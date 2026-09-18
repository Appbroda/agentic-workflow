"""An out-of-scope diff is put in front of the reviewer (87- Part C).

AB-Feature-218's frontend was approved with zero findings. Its diff added
`.catch(() => null)` to `refetchData()` and `refetchPublishersData()` -- two effects that
predate the feature and have nothing to do with bulk CSV upload, so a failed app-list load
now produces no error and no signal at all. It also loosened a query-match branch and added a
pagination guard. All five edits existed to satisfy the model's own new test mocks and came
out of the in-attempt source-repair pass. The review's `architecture_assessment` mentions
none of them.

Part C computes two facts and turns them into one question. Neither is a finding, neither has
a severity, and nothing here can fail a review that would otherwise pass: the reviewer is
asked, and what it concludes is the reviewer's. The second fact is where the spec's wording
had to be made computable -- "hunks that touch lines the change did not otherwise need" is
not knowable, so what is asked is how many disjoint regions of an assigned file moved.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from adapters.llm_adapter import MockCodingExecutor
from agents.engineer.agent import EngineerAgent
from agents.reviewer.agent import (
    _WIDE_CHANGE_HUNK_COUNT,
    ReviewerAgent,
    _declared_scope_areas,
    _within_declared_scope,
)
from prompts.prompt_loader import PromptLoader
from tests.fixtures import commit_all, init_git_repository, start_working_branch
from tests.test_agents import (
    StaticLLMClient,
    StaticValidationTool,
    agent_state,
    domain_payload,
    review_artifact,
    task_plan_artifact,
    technical_prd_artifact,
    validation_result,
)

_ASSIGNED = "src/pages/AllApps.js"
_UNDECLARED = "src/components/Header.js"


def _reviewer(client: StaticLLMClient) -> ReviewerAgent:
    """One reviewer with passing validation, the way the agent unit tests build one."""
    return ReviewerAgent(
        prompt_loader=PromptLoader(),
        llm_client=client,
        validation_tool=StaticValidationTool(
            ruff_result=validation_result("ruff", return_code=0),
            pytest_result=validation_result("pytest", return_code=0),
        ),
    )


def _plan(areas: list[str]) -> Any:
    """A task plan whose review scope declares exactly these files or areas."""
    plan = task_plan_artifact()
    plan.metadata["review_scope"] = {
        "repository_id": "repository-1",
        "workstream_id": "workstream-1",
        "role": "frontend",
        "requirement_ids": ["requirement-1"],
        "expected_files_or_areas": areas,
    }
    return plan


class _CapturingPromptLoader(PromptLoader):
    """The real loader, additionally keeping the context each render was given."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def render(self, template_name: str, **context: Any) -> str:
        self.calls.append((template_name, dict(context)))
        return super().render(template_name, **context)


_ROW_COUNT = 40


def _rows() -> list[str]:
    """The committed body of the assigned file: forty independent one-line exports."""
    return [f"export const row{index:02d} = {index};" for index in range(_ROW_COUNT)]


def _committed_checkout(root: Path) -> Path:
    """A real checkout whose lineage base already holds the assigned file.

    A region count is a diff against that base, so the file has to exist there: a file the
    change *added* has no regions to justify, every line of it being the change.
    """
    init_git_repository(root)
    (root / "src" / "pages").mkdir(parents=True)
    (root / _ASSIGNED).write_text("\n".join(_rows()) + "\n", encoding="utf-8")
    commit_all(root, "checkout baseline")
    start_working_branch(root, "workflow/workflow-1")
    return root


def _edit_every_fifth_row(count: int) -> str:
    """Return the file with ``count`` rows edited, spaced so no two hunks can merge."""
    rows = _rows()
    for index in range(0, count * 5, 5):
        rows[index] = f"export const row{index:02d} = {index} + 1;"
    return "\n".join(rows) + "\n"


async def _review(tmp_path: Path, plan: Any, updates: dict[str, str]) -> tuple[str, str]:
    """Run an attempt then its review, and return the reviewer's instructions and input."""
    completion = (
        await EngineerAgent(
            prompt_loader=PromptLoader(),
            coding_executor=MockCodingExecutor(file_updates=updates),
        ).run(agent_state(tmp_path, [plan]))
    )["artifacts"][0]
    client = StaticLLMClient(domain_payload(review_artifact()))
    update = await _reviewer(client).run(
        agent_state(tmp_path, [technical_prd_artifact(), plan, completion])
    )
    assert update["artifacts"][0].verdict == "approved"
    return client.calls[0]


# --------------------------------------------------------------------------------------
# The membership rule
# --------------------------------------------------------------------------------------


def test_a_declared_token_is_a_file_or_a_directory_and_never_a_guess() -> None:
    """Exact membership, the same discipline `assigned_file_conformance` uses.

    An area this rule cannot interpret makes the question quieter, never wrong. Nothing here
    fuzzy-matches, so a change cannot be asked about a path because it looked similar to one.
    """
    areas = _declared_scope_areas(["./src/apiUtils/", "src/pages/AllApps.js", "", "  "])

    assert areas == ("src/apiUtils", "src/pages/AllApps.js")
    assert _within_declared_scope("src/apiUtils/allapps.apiUtils.js", areas)
    assert _within_declared_scope("src/pages/AllApps.js", areas)
    # A sibling directory whose name merely starts the same way is not inside the area.
    assert not _within_declared_scope("src/apiUtilsLegacy/x.js", areas)
    assert not _within_declared_scope("src/pages/AllAppsHeader.js", areas)
    assert not _within_declared_scope("src/components/Header.js", areas)


# --------------------------------------------------------------------------------------
# The question, and its absence
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_path_outside_the_declared_scope_is_named_in_the_prompt(tmp_path: Path) -> None:
    """The 218 question in its simplest form: this file was not what you were asked for."""
    instructions, input_text = await _review(
        tmp_path,
        _plan([_ASSIGNED]),
        {_ASSIGNED: "export const page = () => null;\n", _UNDECLARED: "export const h = 1;\n"},
    )

    assert "touches behaviour the workstream did not declare" in instructions
    assert f"`{_UNDECLARED}` (production) is outside `expected_files_or_areas`" in instructions
    # The assigned file is not accused of being unassigned.
    assert f"`{_ASSIGNED}` (production) is outside" not in instructions
    evidence = json.loads(input_text)["workspace_change_evidence"]
    assert evidence["out_of_scope_changes"]["undeclared_paths"] == [
        {"path": _UNDECLARED, "change_kind": "production"}
    ]


@pytest.mark.asyncio
async def test_a_diff_entirely_within_scope_renders_a_byte_identical_prompt(
    tmp_path: Path,
) -> None:
    """A change inside what the plan declared pays nothing for this clause -- exactly.

    Asserted by re-rendering the reviewer's own captured context twice: once as the reviewer
    rendered it, and once with Part C's two variables removed, which is the template as it
    stood before this part existed. Byte equality between those two is the claim; searching
    the text for an absent sentence would pass just as well against a clause that leaked a
    blank line.
    """
    loader = _CapturingPromptLoader()
    completion = (
        await EngineerAgent(
            prompt_loader=PromptLoader(),
            coding_executor=MockCodingExecutor(
                file_updates={_ASSIGNED: "export const page = () => null;\n"}
            ),
        ).run(agent_state(tmp_path, [_plan([_ASSIGNED])]))
    )["artifacts"][0]
    client = StaticLLMClient(domain_payload(review_artifact()))
    await ReviewerAgent(
        prompt_loader=loader,
        llm_client=client,
        validation_tool=StaticValidationTool(
            ruff_result=validation_result("ruff", return_code=0),
            pytest_result=validation_result("pytest", return_code=0),
        ),
    ).run(agent_state(tmp_path, [technical_prd_artifact(), _plan([_ASSIGNED]), completion]))

    template, context = next(call for call in loader.calls if call[0] == "reviewer/v1.jinja2")
    assert context["undeclared_change_paths"] == []
    assert context["widely_changed_paths"] == []
    today = PromptLoader().render(
        template,
        **{
            key: value
            for key, value in context.items()
            if key not in {"undeclared_change_paths", "widely_changed_paths"}
        },
    )

    assert PromptLoader().render(template, **context) == today
    assert "touches behaviour the workstream did not declare" not in today


@pytest.mark.asyncio
async def test_nothing_is_asked_when_the_workstream_declared_no_scope(tmp_path: Path) -> None:
    """Nothing was declared, so nothing is outside it. The alternative -- every path of
    every legacy-scoped workflow named as undeclared -- would be noise, not a question."""
    instructions, input_text = await _review(
        tmp_path, _plan([]), {_UNDECLARED: "export const h = 1;\n"}
    )

    assert "is outside `expected_files_or_areas`" not in instructions
    evidence = json.loads(input_text)["workspace_change_evidence"]
    assert evidence["out_of_scope_changes"] == {"undeclared_paths": [], "widely_changed_paths": []}


@pytest.mark.asyncio
async def test_a_test_file_is_named_with_its_kind_rather_than_filtered(tmp_path: Path) -> None:
    """A test beside an assigned file is ordinary suite organisation, and a plan naming only
    production areas would make every such test 'out of scope'. The kind travels with the
    path so the reviewer can weigh that itself, instead of this check deciding for it."""
    test_path = "src/pages/AllApps.test.js"
    instructions, _ = await _review(
        tmp_path,
        _plan(["src/apiUtils"]),
        {
            "src/apiUtils/allapps.apiUtils.js": "export const bulk = () => null;\n",
            test_path: "test('bulk', () => { expect(1).toBe(1); });\n",
        },
    )

    assert f"`{test_path}` (test) is outside `expected_files_or_areas`" in instructions


@pytest.mark.asyncio
async def test_an_assigned_file_changed_in_many_places_is_asked_about_by_region(
    tmp_path: Path,
) -> None:
    """218's actual shape: the incidental edits were *inside* the assigned file.

    So the undeclared-path list finds nothing here, and the region count is the whole of what
    Part C can say. The file is committed first and then edited in scattered places, because
    a region count is a diff against the lineage base and a newly added file has no regions --
    every line of it is the change.
    """
    root = _committed_checkout(tmp_path / "checkout")
    # Edits spread far enough apart that `--unified=0` cannot merge them into one hunk.
    edited = _edit_every_fifth_row(_WIDE_CHANGE_HUNK_COUNT)

    instructions, input_text = await _review(root, _plan([_ASSIGNED]), {_ASSIGNED: edited})

    evidence = json.loads(input_text)["workspace_change_evidence"]
    question = evidence["out_of_scope_changes"]
    assert question["undeclared_paths"] == []
    assert question["widely_changed_paths"] == [
        {"path": _ASSIGNED, "changed_region_count": _WIDE_CHANGE_HUNK_COUNT}
    ]
    assert (
        f"`{_ASSIGNED}` is assigned, and changed in {_WIDE_CHANGE_HUNK_COUNT} separate regions"
        in instructions
    )


@pytest.mark.asyncio
async def test_an_assigned_file_changed_in_few_places_is_not_asked_about(tmp_path: Path) -> None:
    """The counterpart, so a helper that always answers 'many' cannot satisfy both tests.

    One region below the threshold, from the same fixture shape and the same code path.
    """
    root = _committed_checkout(tmp_path / "checkout")
    edited = _edit_every_fifth_row(_WIDE_CHANGE_HUNK_COUNT - 1)

    instructions, input_text = await _review(root, _plan([_ASSIGNED]), {_ASSIGNED: edited})

    evidence = json.loads(input_text)["workspace_change_evidence"]
    assert evidence["out_of_scope_changes"]["widely_changed_paths"] == []
    assert "separate regions" not in instructions


@pytest.mark.asyncio
async def test_the_question_is_durable_on_the_review_artifact(tmp_path: Path) -> None:
    """A question is only worth asking if somebody can later check that it was answered."""
    completion = (
        await EngineerAgent(
            prompt_loader=PromptLoader(),
            coding_executor=MockCodingExecutor(
                file_updates={
                    _ASSIGNED: "export const page = () => null;\n",
                    _UNDECLARED: "export const h = 1;\n",
                }
            ),
        ).run(agent_state(tmp_path, [_plan([_ASSIGNED])]))
    )["artifacts"][0]
    client = StaticLLMClient(domain_payload(review_artifact()))

    update = await _reviewer(client).run(
        agent_state(tmp_path, [technical_prd_artifact(), _plan([_ASSIGNED]), completion])
    )

    review = update["artifacts"][0]
    assert review.metadata["undeclared_change_paths"] == [
        {"path": _UNDECLARED, "change_kind": "production"}
    ]
    assert review.metadata["wide_change_region_threshold"] == _WIDE_CHANGE_HUNK_COUNT
    # The safety invariant, on the one part of this item that touches every review's prompt:
    # the question adds no finding, no limitation and no severity.
    assert review.verdict == "approved"
    assert review.findings == []
    assert review.metadata["manual_review_required"] is False
