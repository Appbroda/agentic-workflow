"""One repository can still disagree with the contract (87- Part B).

`if len(changes) < 2: return _NOT_REVIEWED` was reasoned as "one repository's change cannot
disagree with another's". True, and the wrong conclusion: it can still disagree with the
contract. So a feature where one required workstream never lands got no cross-repository check
of any kind, while the workstream that *did* land was still offered for publication --
AB-Feature-218's terminal message reads "Ready to open on request: AB-console-admin-2.0",
with `cross_repository_behaviour: not_checked` and not one line of its source read by this
gate.

Part B reads that one repository against the contract alone and records
`checked_against_contract` -- a third value, because it is a weaker claim than the
two-repository review and the artifact must not overstate it.

**This does not catch 218, and the last test here says so in the strongest terms available.**
The frontend's `url: '/api/apps/bulk'` sits beside a contract stating the path is
`/api/apps/bulk`: they agree. The disagreement is with an unchanged file that appears in no
diff, which is Part A's job and nothing else's. Part B is a real gap, separately.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from agents.integration_reviewer.agent import (
    IntegrationReviewerAgent,
    assessment_coverage_limitations,
)
from prompts.prompt_loader import PromptLoader
from tests.test_integration_reviewer import (
    SeamLLMClient,
    StubDiffs,
    child_result,
    contract_artifact,
)
from tools.cross_repository_diffs import RepositoryChange, bounded_repository_change

_ABSENT = "backend"
_PRESENT = "frontend"


def _reviewer(payload: dict[str, Any], diffs: Any = None) -> tuple[Any, Any]:
    """Return a reviewer whose seam capability is wired to fakes, and the fake model."""
    client = SeamLLMClient(payload)
    return (
        IntegrationReviewerAgent(
            llm_client=cast(Any, client),
            diff_provider=diffs or StubDiffs(_PRESENT),
            prompt_loader=PromptLoader(),
        ),
        client,
    )


async def _review(reviewer: Any) -> Any:
    """Run the gate for a feature where only the frontend produced a result."""
    return await reviewer.review(
        feature_id="feature-contract",
        contract=contract_artifact(),
        child_results=[child_result(_PRESENT, "approved", True)],
        merge_order=[_PRESENT],
    )


# --------------------------------------------------------------------------------------
# The split
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_one_repository_is_read_against_the_contract_alone() -> None:
    """The review runs, and the artifact makes the weaker claim rather than the stronger one."""
    reviewer, client = _reviewer(
        {"compatibility_assessment": "The change matches the sections it declares.", "findings": []}
    )

    review = await _review(reviewer)

    assert len(client.calls) == 1
    # The gate's deterministic half already blocks a half-landed feature: the contract names
    # `backend` as an owner and no backend result exists, so `missing-owner-backend` is
    # raised whatever this seam review says. Part B does not add that and does not need to --
    # what it adds is that the repository which *did* land was read at all.
    assert [item.finding_id for item in review.cross_repository_findings] == [
        f"missing-owner-{_ABSENT}"
    ]
    coverage = review.metadata["assessment_coverage"]
    assert coverage["cross_repository_behaviour"] == "checked_against_contract"
    # Not the two-repository claim, and it says which one it is.
    assert "was read and compared" not in " ".join(review.contract_checks)
    assert "read against the contract alone" in " ".join(review.contract_checks)
    assert review.metadata["prompt_template"] == (
        "integration_reviewer/single_repository_v1.jinja2"
    )
    assert review.metadata["seam_reviewed_repository_ids"] == [_PRESENT]


@pytest.mark.asyncio
async def test_the_new_value_reads_as_a_limitation_and_never_as_coverage() -> None:
    """`assessment_coverage_limitations` is the one place this value could pass for `checked`.

    It filtered on `== "not_checked"` before, so a third value would have silently dropped out
    of the feature completion summary and read as covered.
    """
    reviewer, _ = _reviewer({"compatibility_assessment": "Conforms.", "findings": []})

    limitations = assessment_coverage_limitations(await _review(reviewer))

    assert any("did not review cross-repository behaviour" in item for item in limitations)
    assert any("read against the contract alone" in item for item in limitations)
    assert any(
        "nothing here checked how the repositories behave together" in item for item in limitations
    )


@pytest.mark.asyncio
async def test_a_finding_naming_the_absent_repository_is_dropped_not_raised() -> None:
    """The contract names the absent repository on nearly every page, so this is expected.

    The two-repository path still raises -- a name outside a set of two means the model
    invented a repository -- but raising here would discard the whole review over a slip
    about scope and return this path to reading nothing, which is what Part B exists to end.
    """
    reviewer, _ = _reviewer(
        {
            "compatibility_assessment": "The provider is missing.",
            "findings": [
                {
                    "finding_id": "PROVIDER_ENDPOINT_ABSENT",
                    "severity": "critical",
                    "responsible_repository_id": _ABSENT,
                    "affected_repository_ids": [_ABSENT, _PRESENT],
                    "contract_reference": "createLogin",
                    "description": "The backend never implemented the endpoint.",
                    "evidence": "not visible in the changed source",
                    "recommended_fix": "Implement it.",
                },
                {
                    "finding_id": "REQUEST_FIELD_NOT_IN_CONTRACT",
                    "severity": "medium",
                    "responsible_repository_id": _PRESENT,
                    "affected_repository_ids": [_PRESENT],
                    "contract_reference": "createLogin",
                    "description": "The change posts a field the contract does not define.",
                    "evidence": "contract: {email}; change: body.user",
                    "recommended_fix": "Send `email`.",
                },
            ],
        }
    )

    review = await _review(reviewer)

    ids = [item.finding_id for item in review.cross_repository_findings]
    assert "PROVIDER_ENDPOINT_ABSENT" not in ids
    assert "REQUEST_FIELD_NOT_IN_CONTRACT" in ids


@pytest.mark.asyncio
async def test_the_prompt_forbids_speculating_about_the_repository_that_did_not_land() -> None:
    """The prompt is the first defence and the filter is the second; both are load-bearing."""
    reviewer, client = _reviewer({"compatibility_assessment": "Conforms.", "findings": []})

    await _review(reviewer)

    [instructions] = client.calls
    assert "Do not speculate about the repository that produced no change" in instructions
    assert f"must be `{_PRESENT}` on every finding" in instructions
    assert "Do not report that the other side is missing" in instructions
    # And the narrower question, in the words the item requires.
    assert "sections of the contract it declares it implements or consumes" in instructions


# --------------------------------------------------------------------------------------
# The honest-scope test
# --------------------------------------------------------------------------------------


class _BulkUploadDiffs:
    """AB-Feature-218's frontend change, as this gate would have been given it.

    The two blockers are absent from it because they were absent from the diff: they live in
    `src/config/axios.js`, which the change does not touch. That is the whole point.
    """

    async def collect(self, *, feature_id: str, child_results: Any) -> list[RepositoryChange]:
        """Return only the frontend, exactly as 218 produced only the frontend."""
        del feature_id, child_results
        return [
            bounded_repository_change(
                repository_id=_PRESENT,
                role="frontend",
                contract_sections_implemented=[],
                contract_sections_consumed=["bulkCreateAppsFromCsv"],
                files=[
                    (
                        "src/apiUtils/allapps.apiUtils.js",
                        "export const bulkCreateAppsFromCsv = (csvFile) => ({\n"
                        "  url: '/api/apps/bulk',\n"
                        "  method: 'POST',\n"
                        "  data: toFormData(csvFile),\n"
                        "});\n",
                    )
                ],
            )
        ]


@pytest.mark.asyncio
async def test_the_218_frontend_and_contract_yield_no_finding_about_the_url() -> None:
    """Part B is not the fix for 218, and this asserts the limitation rather than stating it.

    The contract says the path is `/api/apps/bulk`. The change says `url: '/api/apps/bulk'`.
    Read against each other they agree, and they *do* agree -- the defect is that the
    configured `baseURL` already ends in `/api`, so the effective URL doubles it. That fact
    is not in this evidence and cannot be: `src/config/axios.js` is unchanged and appears in
    no diff. So the assertions below are about what the reviewer was given, not about what a
    model happened to answer -- a stub returning no findings would satisfy "no finding about
    the URL" whatever the prompt contained.
    """
    reviewer, client = _reviewer(
        {"compatibility_assessment": "The declared request matches the contract.", "findings": []},
        _BulkUploadDiffs(),
    )

    review = await _review(reviewer)

    [instructions] = client.calls
    [change] = await _BulkUploadDiffs().collect(feature_id="feature-contract", child_results=[])
    # What the reviewer could see: the change's own URL, and nothing about the base it
    # resolves against.
    assert "/api/apps/bulk" in instructions
    assert "baseURL" not in instructions
    assert "src/config/axios.js" not in instructions
    assert "interceptor" not in instructions
    assert [item["path"] for item in change.files] == ["src/apiUtils/allapps.apiUtils.js"]
    # And so: not one finding about the URL. The only finding on this artifact is the
    # deterministic one about the workstream that never landed, which predates this item and
    # is not about the change at all. Part A is the only part of 87- that catches this run.
    assert [item.finding_id for item in review.cross_repository_findings] == [
        f"missing-owner-{_ABSENT}"
    ]
    assert not any(
        "/api/apps/bulk" in f"{item.description} {item.evidence} {item.recommended_fix}"
        for item in review.cross_repository_findings
    )
    assert (
        review.metadata["assessment_coverage"]["cross_repository_behaviour"]
        == "checked_against_contract"
    )
