"""The three judges of contract compliance are shown the contract they judge against.

Regression cover for 00-todo item 25, whose evidence is run 197's frontend attempt-2. The
independent Reviewer blocked at high severity because `BulkImportApps.js` treated an import as
successful only when `status === 201 && data.code === 'BULK_IMPORT_CREATED' && data.created !==
0`, calling the `code` and non-zero `created` conditions "an additional hard-coded code and
nonzero-created condition ... contrary to the scoped 201 success behavior". The approved
contract's `BulkImportSuccess` makes `code` a `const BULK_IMPORT_CREATED`, `created` an integer
at `minimum: 1`, and both `required`; every zero-created shape in the contract carries a 4xx or
5xx status. The rejected attempt was the contract restated in code, and attempt-3's remediation
removed the check and shipped a client that accepts a 201 the contract rules out.

Nothing about the contract was unavailable. It is a persisted artifact carried in every child's
state and the section names to select by were already in the review scope; the Engineer, its
self-review pass and the Reviewer were simply given the *names* -- `contract_sections_consumed:
[..., "BulkImportSuccess", ...]` -- and asked about compliance. The finding's "undocumented
fields" were inferred from their absence in the reviewer's own context.

So these tests assert two different things, and the distinction matters:

- **What the judges hold.** A model's verdict cannot be asserted. What can be asserted, and is
  the whole mechanism of the fix, is that the facts the -197 finding contradicted are in each
  judge's own prompt: `code` and `created` both listed in `required`, the `const`, and the
  `minimum: 1`. A judge holding that text has no honest route to -197's finding, and the
  negative control below shows that without the selection it holds only the section's name.
- **How the selection behaves at its edges.** Bounded per section and in total, omission
  reported and never a silent trim, and a scope naming a section the contract lacks reported as
  absent rather than dropped.

The contract is the real one, loaded from `tests/fixtures/feature_197_integration_contract.json`
through `IntegrationContractArtifact.model_validate_json`, so a field this repository renamed or
dropped fails at load rather than at an assertion written to agree with it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from adapters.llm_adapter import LLMClient, MockCodingExecutor
from agents.engineer.agent import EngineerAgent
from agents.reviewer.agent import ReviewerAgent
from agents.shared.contract_sections import (
    contract_reference_resolves,
    contract_section_context,
    contract_section_names,
)
from artifacts.schemas import (
    IntegrationContractArtifact,
    ReviewArtifact,
    TaskPlanArtifact,
)
from prompts.prompt_loader import PromptLoader
from tests.test_agents import (
    StaticValidationTool,
    agent_state,
    code_completion_artifact,
    task_plan_artifact,
    technical_prd_artifact,
    validation_result,
)
from tests.test_self_review import ScriptedTextClient, StubSelfReviewer
from tools.self_review import SELF_REVIEW_DIRECTIVE, CodingRoleSelfReviewer

FIXTURES = Path(__file__).parent / "fixtures"

# Run 197's frontend workstream, `ws-ab-console-admin-bulk-ui`, exactly as
# `010_repository_execution_plan.json` recorded it: sixteen names spanning two endpoints, five
# shared schemas and nine error contracts, all consumed and none implemented. The largest real
# selection on record, and the one the blocking review was given only the names of.
FRONTEND_SECTIONS_CONSUMED = [
    "bulkImportApps",
    "downloadAppBulkImportTemplate",
    "AppBulkImportCsvFile",
    "AppBulkImportTemplate",
    "BulkImportSuccess",
    "BulkImportError",
    "AuthorizationError",
    "BULK_IMPORT_FILE_INVALID",
    "BULK_IMPORT_ROW_LIMIT_EXCEEDED",
    "BULK_IMPORT_VALIDATION_FAILED",
    "BULK_IMPORT_DUPLICATE",
    "BULK_IMPORT_FILE_TOO_LARGE",
    "BULK_IMPORT_FAILED",
    "AUTHENTICATION_REQUIRED",
    "APP_MANAGEMENT_FORBIDDEN",
    "BULK_IMPORT_CONTRACT_UNAVAILABLE",
]


def feature_197_contract(*, workflow_id: str = "workflow-1") -> IntegrationContractArtifact:
    """Read run 197's approved contract the way the feature store reads a persisted one.

    `model_validate_json` over the saved bytes, which is exactly `_artifact_from_model`'s own
    deserialization: the artifact's strict model is the gate, so a field this repository has
    since renamed or dropped fails here at load rather than at an assertion written to agree
    with it. Only `workflow_id` is rewritten, because `require_artifact` refuses an artifact
    belonging to a different workflow and the test states its own.
    """
    payload = json.loads((FIXTURES / "feature_197_integration_contract.json").read_text("utf-8"))
    payload["workflow_id"] = workflow_id
    return IntegrationContractArtifact.model_validate_json(json.dumps(payload))


def frontend_task_plan(
    *,
    implemented: list[str] | None = None,
    consumed: list[str] | None = None,
) -> TaskPlanArtifact:
    """Return a task plan carrying 197's frontend review scope, sections included."""
    plan = task_plan_artifact()
    return plan.model_copy(
        update={
            "metadata": {
                **plan.metadata,
                "contract_version": "1.0.0",
                "review_scope": {
                    "repository_id": "AB-console-admin-2.0",
                    "workstream_id": "ws-ab-console-admin-bulk-ui",
                    "role": "frontend",
                    "requirement_ids": ["requirement-1"],
                    "scoped_requirements": [
                        {
                            "requirement_id": "requirement-1",
                            "responsibility": "consumes",
                            "acceptance_criterion_ids": ["criterion-1"],
                        }
                    ],
                    "out_of_scope_requirements": [],
                    "shared_requirements": [],
                    "responsibilities": ["Add the bulk import control."],
                    "acceptance_criteria": [
                        "A 201 response displays processed and created counts and triggers the "
                        "existing AllApps refresh path"
                    ],
                    "test_requirements": ["Cover the success and failure renderings."],
                    "contract_sections_consumed": (
                        FRONTEND_SECTIONS_CONSUMED if consumed is None else consumed
                    ),
                    "contract_sections_implemented": implemented or [],
                    "expected_files_or_areas": ["src/pages/BulkImportApps.js"],
                    "implementation_expectations": [],
                },
            }
        }
    )


# --------------------------------------------------------------------------------------
# Run 197: what each judge now holds
# --------------------------------------------------------------------------------------

# The three facts -197's finding contradicted. A judge holding all three cannot call `code` and
# a non-zero `created` undocumented additions without contradicting the text in front of it.
_CONST_CODE = '"const": "BULK_IMPORT_CREATED"'
_CREATED_MINIMUM = '"minimum": 1'


def _self_review_payload() -> str:
    """One clean self-review response, so the pass records `clean` and the attempt completes."""
    return json.dumps(
        {
            "summary": "The success gate restates the contract's own required fields.",
            "requirement_coverage": [
                {
                    "requirement_id": "requirement-1",
                    "status": "implemented",
                    "evidence": "the 201 branch checks code and created",
                }
            ],
            "findings": [],
        }
    )


def _bulk_import_success(context: dict[str, Any]) -> dict[str, Any]:
    """Return the quoted `BulkImportSuccess` entry from a selection."""
    return next(item for item in context["sections"] if item["name"] == "BulkImportSuccess")


def _review_response(findings: list[dict[str, Any]]) -> str:
    """One schema-valid review over 197's frontend scope, carrying these findings."""
    return json.dumps(
        {
            "verdict": "changes_requested" if findings else "approved",
            "summary": "The success gate restates the contract's own required fields.",
            "requirement_checks": [
                {
                    "requirement_id": "requirement-1",
                    "passed": not findings,
                    "evidence": "code and created are both required by the contract",
                }
            ],
            "findings": findings,
            "architecture_assessment": "Consistent with the contract.",
            "security_assessment": "No new surface.",
            "test_coverage_assessment": "Success and failure paths covered.",
        }
    )


async def _run_reviewer(tmp_path: Path, client: ScriptedTextClient) -> ReviewArtifact:
    """Review 197's frontend workstream with its own contract in state, and return the artifact."""
    update = await ReviewerAgent(
        prompt_loader=PromptLoader(),
        llm_client=cast(LLMClient, client),
        validation_tool=StaticValidationTool(
            ruff_result=validation_result("ruff", return_code=0),
            pytest_result=validation_result("pytest", return_code=0),
        ),
    ).run(
        agent_state(
            tmp_path,
            [
                technical_prd_artifact(),
                feature_197_contract(),
                frontend_task_plan(),
                code_completion_artifact(),
            ],
        )
    )
    return cast(ReviewArtifact, update["artifacts"][0])


async def _review_with_findings(tmp_path: Path, findings: list[dict[str, Any]]) -> ReviewArtifact:
    """Return the published review for a model response carrying these findings."""
    return await _run_reviewer(tmp_path, ScriptedTextClient([_review_response(findings)]))


def _rendered_sections(prompt: str) -> dict[str, Any]:
    """Parse the contract block back out of a rendered prompt, as the judge reads it."""
    # The last occurrence: the rules above the block name it too, and the rule's own text is
    # followed by the review scope's JSON rather than the block's.
    header = "Contract sections in scope"
    body = prompt[prompt.rindex(header) :]
    decoded, _end = json.JSONDecoder().raw_decode(body[body.index("{") :])
    return cast(dict[str, Any], decoded)


def test_the_197_success_schema_is_selected_whole_with_both_conditions_it_states() -> None:
    """The section the review named is quoted, and it states both things the review denied."""
    context = contract_section_context(
        contract=feature_197_contract(), task_plan=frontend_task_plan()
    )
    assert context is not None
    schema = _bulk_import_success(context)["definition"]["json_schema"]

    assert schema["required"] == ["code", "processed", "created", "outcomes"]
    assert schema["properties"]["code"] == {"const": "BULK_IMPORT_CREATED"}
    assert schema["properties"]["created"]["minimum"] == 1
    # The other half of the same fact: the contract has no zero-created success at all, so
    # `created >= 1` is not an extra condition the client invented on top of a 201.
    error = next(item for item in context["sections"] if item["name"] == "BulkImportError")
    assert error["definition"]["json_schema"]["properties"]["created"] == {"const": 0}
    assert all(
        item["definition"]["status_code"] >= 400
        for item in context["sections"]
        if item["kind"] == "error_contract"
    )


def test_every_section_of_197s_frontend_scope_fits_inside_the_bounds() -> None:
    """The largest real selection on record is quoted whole: nothing absent, nothing omitted."""
    context = contract_section_context(
        contract=feature_197_contract(), task_plan=frontend_task_plan()
    )
    assert context is not None
    assert [item["name"] for item in context["sections"]] == FRONTEND_SECTIONS_CONSUMED
    assert context["sections_absent"] == []
    assert context["sections_omitted"] == []
    assert context["characters_selected"] < context["character_bounds"]["total"]
    assert {item["relationship"] for item in context["sections"]} == {"consumes"}


@pytest.mark.asyncio
async def test_the_reviewer_prompt_carries_the_contract_it_may_block_on(
    tmp_path: Path,
) -> None:
    """The blocking authority holds the section's text, not only its name."""
    client = ScriptedTextClient([_review_response([])])
    await _run_reviewer(tmp_path, client)

    instructions = client.calls[0][0]
    assert "Contract sections in scope" in instructions
    assert _CONST_CODE in instructions
    assert _CREATED_MINIMUM in instructions
    # Read back out of the rendered prompt rather than trusting the substrings above: what the
    # judge holds is the parsed block, and the -197 finding is refuted by its `required` list.
    quoted = _rendered_sections(instructions)
    assert _bulk_import_success(quoted)["definition"]["json_schema"]["required"] == [
        "code",
        "processed",
        "created",
        "outcomes",
    ]


@pytest.mark.asyncio
async def test_the_engineer_and_its_self_review_hold_the_same_sections(
    tmp_path: Path,
) -> None:
    """One selection, three judges: the self-review reads the Engineer's own instructions."""
    # The real self-reviewer, not a stub, so the assertion covers the concatenation that
    # actually carries the sections into Part E's own request.
    client = ScriptedTextClient([_self_review_payload()])
    await EngineerAgent(
        prompt_loader=PromptLoader(),
        coding_executor=MockCodingExecutor(
            file_updates={"src/pages/BulkImportApps.js": "export default null;\n"}
        ),
        self_reviewer=CodingRoleSelfReviewer(cast(LLMClient, client)),
    ).run(agent_state(tmp_path, [frontend_task_plan(), feature_197_contract()]))

    assert len(client.calls) == 1
    directive = client.calls[0][0]
    assert "Contract sections in scope" in directive
    assert _CONST_CODE in directive
    assert _CREATED_MINIMUM in directive
    # And the directive itself is unchanged: Part E's prompt and its measurement key are
    # untouched by this change, so the next batch's keep/remove re-tally stays comparable.
    assert directive.endswith(SELF_REVIEW_DIRECTIVE)


@pytest.mark.asyncio
async def test_without_the_contract_a_judge_holds_only_the_section_name(
    tmp_path: Path,
) -> None:
    """The negative control: -197's shape, and the exact blindness it was reviewed under."""
    self_reviewer = StubSelfReviewer(None)
    await EngineerAgent(
        prompt_loader=PromptLoader(),
        coding_executor=MockCodingExecutor(
            file_updates={"src/pages/BulkImportApps.js": "export default null;\n"}
        ),
        self_reviewer=self_reviewer,
    ).run(agent_state(tmp_path, [frontend_task_plan()]))

    instructions = self_reviewer.calls[0][0]
    assert "Contract sections in scope" not in instructions
    # The name, and only the name -- run 197's frontend judges held exactly this much.
    assert "BulkImportSuccess" in instructions
    assert _CONST_CODE not in instructions
    assert _CREATED_MINIMUM not in instructions


# --------------------------------------------------------------------------------------
# The bounds, and the two ways the selection must stay honest
# --------------------------------------------------------------------------------------


def test_a_section_too_large_to_quote_is_named_and_never_trimmed() -> None:
    """Omission with a reason, because two thirds of a `required` list reads as the whole list."""
    context = contract_section_context(
        contract=feature_197_contract(),
        task_plan=frontend_task_plan(consumed=["bulkImportApps", "BulkImportSuccess"]),
        section_max_characters=1_500,
    )
    assert context is not None
    assert [item["name"] for item in context["sections"]] == ["BulkImportSuccess"]
    assert context["sections_omitted"] == [
        {
            "name": "bulkImportApps",
            "kind": "endpoint",
            "relationship": "consumes",
            "characters": context["sections_omitted"][0]["characters"],
            "reason": "over_per_section_character_bound",
        }
    ]
    assert context["sections_omitted"][0]["characters"] > 1_500
    # The oversized section's text is nowhere in the rendered block -- not a fragment of it.
    rendered = json.dumps(context)
    assert "multipart/form-data" not in rendered


def test_the_total_bound_costs_only_the_sections_that_did_not_fit() -> None:
    """A later small section is still quoted; nothing after the first overflow is dropped."""
    context = contract_section_context(
        contract=feature_197_contract(),
        task_plan=frontend_task_plan(consumed=["bulkImportApps", "AUTHENTICATION_REQUIRED"]),
        total_max_characters=2_000,
    )
    assert context is not None
    assert [item["name"] for item in context["sections"]] == ["AUTHENTICATION_REQUIRED"]
    assert [item["reason"] for item in context["sections_omitted"]] == [
        "over_total_character_bound"
    ]
    assert context["characters_selected"] <= 2_000


def test_a_section_the_contract_lacks_is_reported_absent_not_dropped() -> None:
    """A scope naming a section this contract version has no definition for is itself a fact."""
    context = contract_section_context(
        contract=feature_197_contract(),
        task_plan=frontend_task_plan(
            implemented=["bulkImportRollback"], consumed=["BulkImportSuccess"]
        ),
    )
    assert context is not None
    assert context["sections_absent"] == [
        {"name": "bulkImportRollback", "relationship": "implements"}
    ]
    assert [item["name"] for item in context["sections"]] == ["BulkImportSuccess"]


def test_a_section_both_implemented_and_consumed_is_labelled_as_both() -> None:
    """One entry, both relationships, quoted once."""
    context = contract_section_context(
        contract=feature_197_contract(),
        task_plan=frontend_task_plan(
            implemented=["BulkImportSuccess"], consumed=["BulkImportSuccess"]
        ),
    )
    assert context is not None
    assert [item["relationship"] for item in context["sections"]] == ["implements_and_consumes"]


def test_no_contract_and_no_sections_both_select_nothing() -> None:
    """Single-repository workflows and unscoped plans render byte-identical prompts to before."""
    assert contract_section_context(contract=None, task_plan=frontend_task_plan()) is None
    assert (
        contract_section_context(contract=feature_197_contract(), task_plan=task_plan_artifact())
        is None
    )
    assert (
        contract_section_context(
            contract=feature_197_contract(), task_plan=frontend_task_plan(consumed=[])
        )
        is None
    )


# --------------------------------------------------------------------------------------
# The reference space a `contract` finding must cite from
# --------------------------------------------------------------------------------------


def test_the_planner_and_the_selection_agree_on_what_the_contract_defines() -> None:
    """One name set behind both, so accepted-by-the-planner means quotable-to-a-judge."""
    contract = feature_197_contract()
    names = contract_section_names(contract)
    assert set(FRONTEND_SECTIONS_CONSUMED) <= names
    assert "bulkImportRollback" not in names


@pytest.mark.asyncio
async def test_an_invented_citation_is_recorded_and_still_blocks(tmp_path: Path) -> None:
    """The decision on item 25's second clause: recorded on the review, not acted on.

    Refusing blocking authority to an unresolved citation would turn some rejections into
    publications, which is a new blocking authority rather than a filter -- 00-todo item 27's
    open design question. It also would not have caught -197, whose `contract_reference:
    BulkImportSuccess` resolved perfectly and was simply wrong about the section's contents.
    So the fact is put on the record, where the next batch can count it, and the finding keeps
    every bit of authority it has today.
    """
    finding = {
        "finding_id": "FR010-INVENTED-SECTION",
        "severity": "high",
        "title": "The partial-success shape is not handled",
        "description": "The client does not render a partial import result.",
        "recommendation": "Render the partial outcome list.",
        "file_path": "src/pages/BulkImportApps.js",
        "line_number": 42,
        "repository_id": "AB-console-admin-2.0",
        "requirement_id": None,
        "contract_reference": "BulkImportPartialSuccess",
        "responsibility": "consumes",
        "validated_revision": "0123456789abcdef",
        "evidence": "no branch handles a partial result",
        "recommended_fix": "Add the partial branch.",
        "finding_category": "contract",
    }
    review = await _review_with_findings(tmp_path, [finding])

    assert review.metadata["unresolved_contract_references"] == [
        {
            "finding_id": "FR010-INVENTED-SECTION",
            "contract_reference": "BulkImportPartialSuccess",
        }
    ]
    # Unchanged authority: the finding still names what it derives from, so every existing
    # blocking rule reads it exactly as it did before this change.
    assert next(
        item for item in review.findings if item.finding_id == "FR010-INVENTED-SECTION"
    ).names_its_scope()


@pytest.mark.asyncio
async def test_a_resolving_citation_leaves_the_record_empty(tmp_path: Path) -> None:
    """And the review says which sections it was shown, by name, without re-quoting them."""
    finding = {
        "finding_id": "FR010-REAL-SECTION",
        "severity": "medium",
        "title": "The created count is not displayed",
        "description": "The success banner omits the created count.",
        "recommendation": "Show created alongside processed.",
        "file_path": "src/pages/BulkImportApps.js",
        "line_number": 42,
        "repository_id": "AB-console-admin-2.0",
        "requirement_id": None,
        "contract_reference": "BulkImportSuccess.created",
        "responsibility": "consumes",
        "validated_revision": "0123456789abcdef",
        "evidence": "the banner renders processed only",
        "recommended_fix": "Add the created count.",
        "finding_category": "contract",
    }
    review = await _review_with_findings(tmp_path, [finding])

    assert review.metadata["unresolved_contract_references"] == []
    quoted = review.metadata["contract_sections_quoted"]
    assert quoted["contract_version"] == "1.0.0"
    assert quoted["quoted"] == FRONTEND_SECTIONS_CONSUMED
    assert quoted["absent"] == []
    assert quoted["omitted"] == []
    # The names and the counts, never a second copy of the schemas on a persisted artifact.
    assert "BULK_IMPORT_CREATED" not in json.dumps(quoted)


def test_a_citation_resolves_when_it_names_a_real_section_however_precisely() -> None:
    """A field path is a citation; only a reference naming no section at all is unresolved."""
    names = contract_section_names(feature_197_contract())
    assert contract_reference_resolves("BulkImportSuccess", names)
    assert contract_reference_resolves("BulkImportSuccess.created", names)
    assert contract_reference_resolves("the 201 shape of bulkImportApps", names)
    assert not contract_reference_resolves("BulkImportPartialSuccess", names)
    assert not contract_reference_resolves(None, names)
    # No contract in the workflow means no reference space to be wrong about.
    assert contract_reference_resolves("anything", frozenset())
