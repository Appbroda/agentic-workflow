"""Cross-repository contract gate that publishes only the checks it actually performed."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from adapters.llm_adapter import LLMClient
from agents.shared.contracts import (
    FEATURE_ARTIFACT_FILENAMES,
    AgentArtifactError,
    create_artifact,
    execution_metadata,
    parse_model_json,
)
from artifacts.schemas import (
    INTEGRATION_BLOCKING_SEVERITIES,
    ChildWorkflowResultArtifact,
    IntegrationContractArtifact,
    IntegrationReviewArtifact,
)
from prompts.prompt_loader import PromptLoader
from state.enums import IntegrationReviewStatus
from tools.contract_tools import ContractCodeGenerator, OpenAPIContractCodeGenerator
from tools.cross_repository_diffs import (
    CrossRepositoryDiffs,
    NullCrossRepositoryDiffs,
    RepositoryChange,
)

# The dimensions this gate reports on, and whether it can reach them at all. It reads child
# *results* and the approved contract; it never opens a repository diff, so everything below
# contract conformance is outside what it can see. Stating them as assessments anyway made an
# approval read as a cross-repository assurance: a consumer calling a provider endpoint with
# the wrong field name would have been called compatible by the one artifact spanning both,
# and child reviews cannot catch it either because each is scoped to its own repository.
_CONTRACT_CONFORMANCE_DIMENSION = "contract_conformance"
_UNREACHABLE_DIMENSIONS = ("cross_repository_behaviour", "security", "deployment")
_DIMENSION_LABELS = {
    "cross_repository_behaviour": "cross-repository behaviour",
    "security": "security",
    "deployment": "deployment",
}
# The three coverage values a dimension can carry, and the reason the third exists. A feature
# where one required workstream never lands used to get no seam review at all, while the
# workstream that *did* land was still offered for publication -- AB-Feature-218's terminal
# message reads "Ready to open on request: AB-console-admin-2.0". One repository's change can
# still disagree with the contract, so it is now read against the contract alone. That is a
# weaker claim than the two-repository review and it gets its own value rather than borrowing
# either neighbour's: `checked` would overstate it and `not_checked` would hide that anything
# was read.
_CHECKED = "checked"
_NOT_CHECKED = "not_checked"
_CHECKED_AGAINST_CONTRACT = "checked_against_contract"
# The one-repository prompt. A separate template, not a conditional in the seam prompt: the
# question is different in kind, and the seam prompt's every instruction is about comparing
# two sides.
_SINGLE_REPOSITORY_TEMPLATE = "integration_reviewer/single_repository_v1.jinja2"


def assessment_coverage_limitations(review: IntegrationReviewArtifact) -> list[str]:
    """Return the operator-facing limitation for whatever this gate could not reach.

    Read back from the artifact's own recorded coverage rather than restated by the caller, so
    a gate that later gains one of these dimensions drops the limitation from the feature
    completion summary without that module changing. An artifact written before coverage was
    recorded yields nothing, which is the same silence it has always had.

    `checked_against_contract` is a limitation here and not coverage. This function is the one
    place the new value could silently read as "checked", so the test is "is it exactly
    `checked`" rather than "is it `not_checked`" -- and a dimension carrying a value this
    module has never heard of is a limitation too, because the alternative is publishing an
    assurance nobody made. A dimension the artifact does not mention at all stays silent, so
    coverage recorded before a dimension existed reads the way it always has.
    """
    coverage = review.metadata.get("assessment_coverage")
    if not isinstance(coverage, dict):
        return []
    limited = {
        dimension: coverage.get(dimension)
        for dimension in _DIMENSION_LABELS
        if isinstance(coverage.get(dimension), str) and coverage.get(dimension) != _CHECKED
    }
    limitations: list[str] = []
    if limited:
        labels = [_DIMENSION_LABELS[dimension] for dimension in limited]
        listed = f"{', '.join(labels[:-1])} or {labels[-1]}" if len(labels) > 1 else labels[0]
        limitations.append(
            f"The integration review checked contract conformance only: it did not review {listed}."
        )
    if _CHECKED_AGAINST_CONTRACT in limited.values():
        limitations.append(
            "One repository's change was read against the contract alone: no second "
            "repository produced a change to compare it with, so nothing here checked how "
            "the repositories behave together."
        )
    return limitations


@dataclass(frozen=True, slots=True)
class _SeamReview:
    """What reviewing the repositories together produced, including having not done it.

    ``coverage`` is the value the artifact records for `cross_repository_behaviour`, carried
    here rather than derived from ``reviewed`` because there are now three answers and
    ``reviewed`` only distinguishes two.
    """

    reviewed: bool
    findings: list[dict[str, Any]]
    assessment: str
    coverage_statement: str
    metadata: dict[str, Any]
    coverage: str = _NOT_CHECKED


_NOT_REVIEWED = _SeamReview(
    reviewed=False,
    findings=[],
    assessment=(
        "Behavioural compatibility between repositories was NOT assessed: no repository diff "
        "was available to this gate, so consumer call sites were not compared against provider "
        "implementations."
    ),
    coverage_statement=(
        "No repository diff was read, so no check here covers the behaviour of the changed source."
    ),
    metadata={},
)


class IntegrationReviewerAgent:
    """Gate coordinated publication on the contract, and on the seam when it can read it.

    Without a diff source and a model this is a contract-conformance gate deciding from child
    result metadata alone, and its artifact says exactly that. Given both, it also reviews the
    changed source of every repository together -- the only point in the platform where more
    than one repository is looked at at once -- and its coverage metadata changes to match.
    """

    def __init__(
        self,
        *,
        contract_generator: ContractCodeGenerator | None = None,
        llm_client: LLMClient | None = None,
        diff_provider: CrossRepositoryDiffs | None = None,
        prompt_loader: PromptLoader | None = None,
    ) -> None:
        """Inject code-generation checks so unit tests remain independent from Node tooling."""
        self._contract_generator = contract_generator or OpenAPIContractCodeGenerator()
        self._llm_client = llm_client
        self._diff_provider = diff_provider or NullCrossRepositoryDiffs()
        self._prompt_loader = prompt_loader

    async def _review_the_seam(
        self,
        *,
        feature_id: str,
        contract: IntegrationContractArtifact,
        child_results: Sequence[ChildWorkflowResultArtifact],
    ) -> _SeamReview:
        """Read every repository's change together, or report that nothing did.

        Returns `_NOT_REVIEWED` whenever the capability is absent or produces nothing to read.
        Silence and a clean result are different claims, and reporting the second for the
        first is the false assurance this gate was corrected for in the first place.

        Three cases, where there used to be two. Nothing to read is still `_NOT_REVIEWED`.
        Two or more repositories are still read together, which is the only check here that
        can find one repository disagreeing with another. *One* repository used to be
        discarded with the same silence as none -- "one repository's change cannot disagree
        with another's" is true and was the wrong conclusion, because it can still disagree
        with the contract, and AB-Feature-218 offered exactly that workstream for publication
        with `cross_repository_behaviour: not_checked` and nothing read at all.
        """
        if self._llm_client is None or self._prompt_loader is None:
            return _NOT_REVIEWED
        changes = await self._diff_provider.collect(
            feature_id=feature_id, child_results=child_results
        )
        if not changes:
            return _NOT_REVIEWED
        if len(changes) == 1:
            return await self._review_against_the_contract(
                feature_id=feature_id,
                contract=contract,
                change=changes[0],
                child_results=child_results,
            )
        rendered = json.dumps(
            [item.model_dump(mode="json") for item in changes], indent=2, sort_keys=True
        )
        instructions = self._prompt_loader.render(
            "integration_reviewer/v1.jinja2",
            feature_id=feature_id,
            contract=contract.model_dump_json(indent=2),
            repository_changes=rendered,
        )
        response = await self._llm_client.respond(instructions=instructions, input_text=rendered)
        payload = parse_model_json(
            response.output_text, expected_keys=("compatibility_assessment", "findings")
        )
        findings = _seam_findings(
            payload,
            {item.repository_id for item in changes},
            {
                item.finding_id
                for item in await self._contract_generator.validate_contract(
                    contract, child_results
                )
            },
        )
        assessment = payload.get("compatibility_assessment")
        truncated = [item.repository_id for item in changes if item.truncated]
        return _SeamReview(
            reviewed=True,
            findings=findings,
            assessment=(
                f"The changed source of {len(changes)} repositories was reviewed together. "
                f"{assessment if isinstance(assessment, str) else ''}"
            ).strip(),
            coverage_statement=(
                f"The changed source of {len(changes)} repositories was read and compared."
                + (
                    f" It did not fit the review budget for: {', '.join(sorted(truncated))}."
                    if truncated
                    else ""
                )
            ),
            metadata={
                "seam_review_model": response.model,
                "seam_review_response_id": response.response_id,
                "seam_reviewed_repository_ids": [item.repository_id for item in changes],
                "seam_review_truncated_repository_ids": sorted(truncated),
                "prompt_template": "integration_reviewer/v1.jinja2",
                **execution_metadata(
                    agent_type="Integration reviewer",
                    provider=response.provider,
                    model=response.model,
                    reasoning_effort=response.reasoning_effort,
                    model_role=response.model_role,
                    model_variable=response.model_variable,
                    routing_reason=response.routing_reason,
                ),
            },
            coverage=_CHECKED,
        )

    async def _review_against_the_contract(
        self,
        *,
        feature_id: str,
        contract: IntegrationContractArtifact,
        change: RepositoryChange,
        child_results: Sequence[ChildWorkflowResultArtifact],
    ) -> _SeamReview:
        """Read the one repository that landed against the contract it declares.

        A narrower question than the seam prompt asks, and the artifact says so: this cannot
        find one repository disagreeing with another, because there is no other change to
        disagree with. AB-Feature-218's two production blockers would not be found here
        either -- its frontend's `url: '/api/apps/bulk'` sits beside a contract that states
        the path is `/api/apps/bulk`, and the disagreement is with an *unchanged* file that
        appears in no diff. That is Part A's job, and the honest-scope test says so.

        A finding naming a repository that produced no change is dropped rather than
        rejected. The two-repository path raises on one, because there a name outside the set
        means the model invented a repository; here the contract itself names the absent
        repository on nearly every page, so a slip is expected and discarding the whole
        review over it would put this path back to reading nothing.
        """
        assert self._llm_client is not None  # noqa: S101 -- checked by the only caller
        assert self._prompt_loader is not None  # noqa: S101
        rendered = json.dumps(change.model_dump(mode="json"), indent=2, sort_keys=True)
        instructions = self._prompt_loader.render(
            _SINGLE_REPOSITORY_TEMPLATE,
            feature_id=feature_id,
            repository_id=change.repository_id,
            contract=contract.model_dump_json(indent=2),
            repository_change=rendered,
        )
        response = await self._llm_client.respond(instructions=instructions, input_text=rendered)
        payload = parse_model_json(
            response.output_text, expected_keys=("compatibility_assessment", "findings")
        )
        findings = _seam_findings(
            payload,
            {change.repository_id},
            {
                item.finding_id
                for item in await self._contract_generator.validate_contract(
                    contract, child_results
                )
            },
            drop_unknown_repository=True,
        )
        assessment = payload.get("compatibility_assessment")
        statement = (
            f"Only {change.repository_id} produced a change, so its source was read against "
            "the contract alone. No comparison against another repository's source was "
            "possible and none was made."
        )
        return _SeamReview(
            reviewed=True,
            findings=findings,
            assessment=(f"{statement} {assessment if isinstance(assessment, str) else ''}").strip(),
            coverage_statement=statement
            + (" Its changed source did not fit the review budget." if change.truncated else ""),
            metadata={
                "seam_review_model": response.model,
                "seam_review_response_id": response.response_id,
                "seam_reviewed_repository_ids": [change.repository_id],
                "seam_review_truncated_repository_ids": (
                    [change.repository_id] if change.truncated else []
                ),
                "prompt_template": _SINGLE_REPOSITORY_TEMPLATE,
                **execution_metadata(
                    agent_type="Integration reviewer",
                    provider=response.provider,
                    model=response.model,
                    reasoning_effort=response.reasoning_effort,
                    model_role=response.model_role,
                    model_variable=response.model_variable,
                    routing_reason=response.routing_reason,
                ),
            },
            coverage=_CHECKED_AGAINST_CONTRACT,
        )

    async def review(
        self,
        *,
        feature_id: str,
        contract: IntegrationContractArtifact,
        child_results: Sequence[ChildWorkflowResultArtifact],
        merge_order: Sequence[str],
    ) -> IntegrationReviewArtifact:
        """Return a versioned approval or repository-targeted correction request."""
        generator_findings = await self._contract_generator.validate_contract(
            contract, child_results
        )
        findings = [
            {
                "finding_id": item.finding_id,
                "severity": item.severity,
                "responsible_repository_id": item.responsible_repository_id,
                "affected_repository_ids": list(item.affected_repository_ids),
                "contract_reference": item.contract_reference,
                "description": item.description,
                "evidence": item.evidence,
                "recommended_fix": item.recommended_fix,
            }
            for item in generator_findings
        ]
        seam = await self._review_the_seam(
            feature_id=feature_id, contract=contract, child_results=child_results
        )
        findings.extend(seam.findings)
        # The write contract for this gate. `medium` blocks here even though the artifact
        # validator still admits it on read, because that validator governs history it cannot
        # rewrite -- see `approved_review_cannot_have_blocking_findings`.
        blocking = [
            item for item in findings if item["severity"] in INTEGRATION_BLOCKING_SEVERITIES
        ]
        blocked = bool(blocking)
        status = (
            IntegrationReviewStatus.CHANGES_REQUESTED
            if blocked
            else IntegrationReviewStatus.APPROVED
        )
        # Attributed to the injected generator rather than describing its individual checks:
        # a richer generator may check more than the default one, and the gate must not
        # narrate coverage it did not choose.
        validator = type(self._contract_generator).__name__
        conformance = (
            f"{validator} reported at least one blocking contract-conformance finding; see "
            "cross_repository_findings."
            if blocked
            else f"{validator} reported no blocking contract-conformance finding."
        )
        return create_artifact(
            IntegrationReviewArtifact,
            workflow_id=feature_id,
            artifact_id=FEATURE_ARTIFACT_FILENAMES["integration_review"],
            producer="integration_reviewer",
            payload={
                "feature_id": feature_id,
                "contract_artifact_id": contract.artifact_id,
                "review_status": status,
                "repository_results": [
                    {
                        "repository_id": item.repository_id,
                        "child_workflow_id": item.child_workflow_id,
                        "status": item.status,
                        "child_result_artifact_id": item.artifact_id,
                    }
                    for item in child_results
                ],
                "contract_checks": [
                    f"Contract-conformance checks were run by {validator} against contract "
                    f"version {contract.contract_version}.",
                    (
                        "The review status is derived only from the severity of the findings "
                        "those checks produced."
                    ),
                    seam.coverage_statement,
                ],
                "cross_repository_findings": findings,
                "compatibility_assessment": f"{conformance} {seam.assessment}",
                "security_assessment": (
                    "NOT PERFORMED. This gate inspects no repository source and makes no "
                    "security claim about the coordinated change. Each repository's own "
                    "007_review.json carries the security assessment for its diff."
                ),
                "deployment_assessment": (
                    "NOT PERFORMED. merge_order below is the approved repository execution "
                    "plan's order filtered to the repositories that are ready; no deployment, "
                    "migration, or rollback verification was carried out against it."
                ),
                "merge_order": list(merge_order),
                # Only what actually blocks. This list used to be built from every finding
                # while the status was decided by severity alone, so an approved review could
                # -- and on AB-Feature-215 did -- carry a populated `required_fixes` that no
                # gate read and nothing acted on. Nothing is lost by narrowing it: every
                # finding keeps its own `recommended_fix` in `cross_repository_findings`,
                # which is where a non-blocking recommendation belongs.
                "required_fixes": [item["recommended_fix"] for item in blocking],
            },
            metadata={
                "source_artifact_ids": [
                    contract.artifact_id,
                    *[item.artifact_id for item in child_results],
                ],
                "contract_version": contract.contract_version,
                "contract_validator": validator,
                # Machine-readable alongside the prose above, so an operator view or a later
                # gate can branch on coverage without parsing an assessment sentence.
                "assessment_coverage": {
                    _CONTRACT_CONFORMANCE_DIMENSION: _CHECKED,
                    **{dimension: _NOT_CHECKED for dimension in _UNREACHABLE_DIMENSIONS},
                    # Whatever the seam review actually claims, which is now one of three
                    # values. Carried on the result rather than derived from `reviewed`,
                    # because `reviewed` cannot tell "read against the contract alone" from
                    # "read against another repository's source".
                    "cross_repository_behaviour": seam.coverage,
                },
                **seam.metadata,
            },
        )


def _seam_findings(
    payload: dict[str, Any],
    repository_ids: set[str],
    existing_ids: set[str],
    *,
    drop_unknown_repository: bool = False,
) -> list[dict[str, Any]]:
    """Keep only findings this gate can stand behind, and route each to a real repository.

    A blocking finding here sends a repository back through its whole engineer/reviewer loop,
    so an invented one costs a rework cycle. Two rules do the filtering: a finding must name a
    repository that is actually in this feature, and it must not collide with an identifier
    the deterministic checks already used.

    ``drop_unknown_repository`` decides what happens to a finding naming a repository outside
    the set, and the two answers are both deliberate. With two or more changes under review,
    a name outside the set means the model invented a repository, and rejecting the response
    is right: the whole review is unreliable. With one change, the contract itself names the
    absent repository on nearly every page, so the model naming it is an expected slip about
    scope rather than a hallucination -- and rejecting the response over one would return
    that path to reading nothing, which is the silence 87- Part B exists to end. Dropping is
    always the less blocking outcome, which is why it is safe to offer at all.
    """
    raw = payload.get("findings")
    if not isinstance(raw, list):
        msg = "integration review response field 'findings' must be a list"
        raise AgentArtifactError(msg)
    findings: list[dict[str, Any]] = []
    seen = set(existing_ids)
    for item in raw:
        if not isinstance(item, dict):
            msg = "integration review finding must be a JSON object"
            raise AgentArtifactError(msg)
        responsible = item.get("responsible_repository_id")
        finding_id = item.get("finding_id")
        if not isinstance(responsible, str) or responsible not in repository_ids:
            if drop_unknown_repository:
                continue
            msg = (
                "integration review finding must name a repository in this feature: "
                f"{sorted(repository_ids)}"
            )
            raise AgentArtifactError(msg)
        if not isinstance(finding_id, str) or not finding_id.strip() or finding_id in seen:
            msg = "integration review findings must carry unique non-empty identifiers"
            raise AgentArtifactError(msg)
        affected = item.get("affected_repository_ids")
        # Narrowed rather than rejected: naming an unrelated repository as affected is a
        # reporting slip, and discarding a real mismatch over it would be the worse trade.
        item["affected_repository_ids"] = [
            value
            for value in (affected if isinstance(affected, list) else [])
            if value in repository_ids
        ] or [responsible]
        seen.add(finding_id)
        findings.append(item)
    return findings
