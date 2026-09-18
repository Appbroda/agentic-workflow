"""Every role that builds or judges a screen is shown the design it is judging against.

The frontend half of the failure `test_contract_sections_for_judges.py` covers for the contract.
A reviewer holding the design can say "this uses a raw hex where the design names a token"; a
reviewer holding nothing has been saying "looks reasonable" for the entire life of this
platform, because prose was the only thing it was ever given.

Five roles, and the assertions split the way that file's do:

- **What each role holds.** A model's verdict cannot be asserted. What can be asserted is that
  the design's own text -- its style names, its component names, its layout, its characters --
  is in each prompt, and that without a snapshot each prompt is exactly what it always was.
- **That the three judges hold the *same* thing.** Byte identity between the Engineer's block
  and the Reviewer's, which is stronger than field equality: the self-review pass is handed the
  Engineer's rendered instructions verbatim, so one selection function on two call sites is the
  whole mechanism, and two selections that merely agree today would not be.
- **How the selection behaves at its edges.** A frame the plan names that the resolution
  omitted reaches the judges as the omission, not as silence; a frame the plan invents is
  refused by the planner rather than reaching an engineer at all.

The snapshot under test is built by the shipped resolver from the live-captured Figma payload in
`tests/fixtures/figma/`, so the design these judges are shown is a real one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from adapters.llm_adapter import LLMClient, MockCodingExecutor
from agents.engineer.agent import EngineerAgent
from agents.planner.feature_planner import (
    DeterministicFeaturePlanner,
    _ensure_plan_names_only_real_design_nodes,
)
from agents.reviewer.agent import ReviewerAgent
from agents.shared.contracts import FEATURE_ARTIFACT_FILENAMES, AgentArtifactError
from agents.shared.design_snapshot import (
    design_detail_in_state,
    design_node_names,
    design_request_context,
    design_snapshot_context,
    design_snapshot_context_from_state,
    scoped_design_node_ids,
)
from artifacts.schemas import (
    DesignDetailArtifact,
    DesignSnapshotArtifact,
    RepositoryExecutionPlanArtifact,
    TaskPlanArtifact,
)
from prompts.prompt_loader import PromptLoader
from services.design_resolution import (
    DeterministicDesignResolver,
    design_content_is_buildable,
)
from tests.test_agents import (
    StaticValidationTool,
    agent_state,
    code_completion_artifact,
    technical_prd_artifact,
    validation_result,
)
from tests.test_contract_sections_for_judges import (
    _review_response,
    feature_197_contract,
    frontend_task_plan,
)
from tests.test_design_snapshot import (
    REAL_FILE_KEY,
    REAL_FRAME_NAME,
    captured_subtree,
    one_file,
    resolver,
)
from tests.test_design_snapshot import _ScriptedFigmaClient as ScriptedFigmaClient
from tests.test_self_review import ScriptedTextClient, StubSelfReviewer

# The frame the captured file actually contains, cited the way a browser writes the link.
REAL_FRAME_NODE_ID = "10:11"
REAL_FRAME_URL = f"https://www.figma.com/design/{REAL_FILE_KEY}/Untitled?node-id=10-11"


async def real_snapshot(*, workflow_id: str = "workflow-1") -> DesignSnapshotArtifact:
    """Resolve the live-captured frame through the shipped resolver, once.

    The design these judges are shown is therefore a real one, extracted by the code that will
    extract it in production, rather than a shape written to suit an assertion.
    """
    from artifacts.design_references import DesignReference

    client = ScriptedFigmaClient(nodes={REAL_FILE_KEY: one_file([captured_subtree()])})
    snapshot = await resolver(client).resolve(
        feature_id=workflow_id,
        references=[DesignReference(url=REAL_FRAME_URL, label="the clock")],
        artifact_id=FEATURE_ARTIFACT_FILENAMES["design_snapshot"],
    )
    return snapshot.model_copy(update={"workflow_id": workflow_id})


async def real_detail(
    *,
    workflow_id: str = "workflow-1",
    node_ids: list[str] | None = None,
    snapshot_artifact_id: str = FEATURE_ARTIFACT_FILENAMES["design_snapshot"],
    **bounds: int,
) -> DesignDetailArtifact:
    """Resolve the live-captured frame at build fidelity, the way a workstream is given it.

    The other half of `real_snapshot`. After 96- the snapshot holds *index* records -- names,
    structure and text -- and this is what an engineer is actually handed, so every assertion
    about what a judge reads has to come through here. Resolved by the shipped resolver from
    the same live capture, for `real_snapshot`'s reason.
    """
    from artifacts.design_references import DesignReference

    wanted = node_ids or [REAL_FRAME_NODE_ID]
    client = ScriptedFigmaClient(
        nodes={REAL_FILE_KEY: one_file([captured_subtree(item) for item in wanted])}
    )
    # The citation names every frame the plan may assign, in the URL spelling a browser writes.
    cited = ",".join(item.replace(":", "-") for item in wanted)
    detail = await resolver(client, **bounds).resolve_detail(
        feature_id=workflow_id,
        repository_id="frontend",
        workstream_id="frontend",
        snapshot_artifact_id=snapshot_artifact_id,
        references=[
            DesignReference(
                url=f"https://www.figma.com/design/{REAL_FILE_KEY}/Untitled?node-id={cited}",
                label="the clock",
            )
        ],
        node_ids=wanted,
        artifact_id=FEATURE_ARTIFACT_FILENAMES["design_detail"],
    )
    return detail.model_copy(update={"workflow_id": workflow_id})


def plan_naming(nodes: list[str]) -> TaskPlanArtifact:
    """197's frontend task plan, with these frames added to its review scope."""
    plan = frontend_task_plan()
    scope = dict(cast("dict[str, Any]", plan.metadata["review_scope"]))
    scope["design_nodes"] = nodes
    return plan.model_copy(update={"metadata": {**plan.metadata, "review_scope": scope}})


# What each role's design block is introduced by. Anchored on the heading rather than on a
# field name, because `design_nodes` legitimately appears in the review scope too -- that is
# the assignment, and the block below the heading is the design. The full parenthetical rather
# than the phrase, because the reviewer's *rules* quote the block's own title, and anchoring on
# the short form would parse the review scope that follows them instead.
ENGINEER_HEADING = "The design this workstream builds (the frames the author attached"
REVIEWER_HEADING = "The design in scope (the frames the author attached"


def rendered_design(prompt: str, heading: str) -> dict[str, Any]:
    """Parse the design block back out of a rendered prompt, as the role reads it."""
    return cast("dict[str, Any]", json.loads(rendered_design_text(prompt, heading)))


def rendered_design_text(prompt: str, heading: str) -> str:
    """The design block's raw rendered characters, exactly as they sit in the prompt.

    Raw rather than re-parsed, so byte identity can be asserted on what the prompts actually
    carry: a `json.dumps(..., sort_keys=True)` round trip normalizes key order and spacing,
    which is precisely the kind of difference "byte identical" exists to rule out.
    """
    body = prompt[prompt.index(heading) :]
    start = body.index("{")
    _decoded, end = json.JSONDecoder().raw_decode(body[start:])
    return body[start : start + end]


# --------------------------------------------------------------------------------------
# What each of the five roles holds
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_product_manager_is_shown_the_whole_design_as_part_of_the_request() -> None:
    """Unscoped, because no workstream exists yet.

    The product manager is deriving the requirements the frames imply; scoping it would be
    scoping by a decision nobody has made.
    """
    snapshot = await real_snapshot()

    context = design_request_context(snapshot)

    assert context is not None
    assert [item["node_id"] for item in context["design_nodes"]] == [REAL_FRAME_NODE_ID]
    assert context["design_nodes"][0]["label"] == "the clock"
    assert context["design_nodes"][0]["design"]["name"] == REAL_FRAME_NAME
    assert context["design_nodes"][0]["source_url"] == REAL_FRAME_URL
    # The bounds and the name source travel with it, so a reader of the prompt knows both what
    # was left out and where the names came from.
    assert context["style_name_source"] == "node_styles"
    assert context["bounds"]["depth"] > 0
    assert design_request_context(None) is None


@pytest.mark.asyncio
async def test_the_product_manager_prompt_carries_the_design_and_its_criteria_rule() -> None:
    """The block, and the paragraph that stops it producing a criterion no diff can satisfy."""
    snapshot = await real_snapshot()
    context = design_request_context(snapshot)

    rendered = PromptLoader().render(
        "product_manager/v1.jinja2",
        workflow_id="workflow-1",
        prd="{}",
        design_snapshot=json.dumps(context, indent=2, sort_keys=True),
    )

    assert "Attached design" in rendered
    assert REAL_FRAME_NAME in rendered
    # The trap the spec names: "matches the mock" is a criterion no reviewer reading a diff can
    # judge, so the prompt has to say what to write instead.
    assert "Matches the mock" in rendered
    assert "pixel fidelity" in rendered
    assert "reused rather than re-implemented" in rendered
    # And the three omission lists are named, so it does not write requirements about frames
    # nobody read.
    for field in ("design_nodes_omitted", "design_nodes_absent", "design_nodes_unreachable"):
        assert field in rendered

    # Without a design, the prompt is exactly what it always was.
    plain = PromptLoader().render(
        "product_manager/v1.jinja2", workflow_id="workflow-1", prd="{}", design_snapshot=""
    )
    assert "Attached design" not in plain
    assert "Matches the mock" not in plain


@pytest.mark.asyncio
async def test_the_planner_prompt_carries_the_design_and_refuses_to_guess_from_role() -> None:
    """The planner assigns, and the prompt says what it may not assign from.

    This platform does not know which repository renders UI. `role` is a descriptive label, and
    a prompt that invited the model to read it that way would encode exactly the assumption the
    repo-agnostic rule forbids.
    """
    snapshot = await real_snapshot()
    context = design_request_context(snapshot)

    rendered = PromptLoader().render(
        "planner/feature_v1.jinja2",
        feature_id="feature-1",
        technical_prd="{}",
        repositories=[],
        has_reconnaissance=False,
        reconnaissance="[]",
        design_snapshot=json.dumps(context, indent=2, sort_keys=True),
    )

    assert "Attached design" in rendered
    assert "design_nodes" in rendered
    assert "Do **not** decide it from a workstream's `role`" in rendered
    assert "does not know which of these repositories renders a user interface" in rendered
    # `applies_to` constrains the assignment where the author set one.
    assert "applies_to" in rendered

    plain = PromptLoader().render(
        "planner/feature_v1.jinja2",
        feature_id="feature-1",
        technical_prd="{}",
        repositories=[],
        has_reconnaissance=False,
        reconnaissance="[]",
        design_snapshot="",
    )
    assert "Attached design" not in plain


@pytest.mark.asyncio
async def test_the_engineer_prompt_carries_the_design_its_own_plan_assigns(
    tmp_path: Path,
) -> None:
    """Scoped to this workstream's frames, and the self-review pass holds the same string."""
    snapshot = await real_snapshot()
    detail = await real_detail(node_ids=[REAL_FRAME_NODE_ID])
    self_reviewer = StubSelfReviewer(None)

    await EngineerAgent(
        prompt_loader=PromptLoader(),
        coding_executor=MockCodingExecutor(
            file_updates={"src/pages/BulkImportApps.js": "export default null;\n"}
        ),
        self_reviewer=self_reviewer,
    ).run(
        agent_state(
            tmp_path,
            [plan_naming([REAL_FRAME_NODE_ID]), feature_197_contract(), snapshot, detail],
        )
    )

    instructions = self_reviewer.calls[0][0]
    assert "The design this workstream builds" in instructions
    assert REAL_FRAME_NAME in instructions
    # Names before values, said in the prompt as well as rendered in the block.
    assert "`color/surface/raised` tells you what this repository already has" in instructions
    assert "design_nodes_omitted" in instructions
    # The self-review pass is handed exactly this string, so it is the same design by
    # construction rather than by a third call.
    assert self_reviewer.calls[0][0] == instructions


@pytest.mark.asyncio
async def test_the_reviewer_prompt_carries_the_design_it_may_block_on(tmp_path: Path) -> None:
    """The blocking authority holds the design, and is told what it may and may not judge."""
    snapshot = await real_snapshot()
    detail = await real_detail(node_ids=[REAL_FRAME_NODE_ID])
    client = ScriptedTextClient([_review_response([])])

    await ReviewerAgent(
        prompt_loader=PromptLoader(),
        llm_client=cast("LLMClient", client),
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
                plan_naming([REAL_FRAME_NODE_ID]),
                code_completion_artifact(),
                snapshot,
                detail,
            ],
        )
    )

    instructions = client.calls[0][0]
    assert "The design in scope" in instructions
    assert REAL_FRAME_NAME in instructions
    # What it may judge, and the thing it may never judge.
    assert "never against an\n  appearance you cannot see" in instructions
    assert "cannot judge pixel\n  fidelity" in instructions
    assert "raw value" in instructions


@pytest.mark.asyncio
async def test_the_three_judges_are_given_byte_identical_design_blocks(
    tmp_path: Path,
) -> None:
    """One selection, byte for byte, across the Engineer, its self-review and the Reviewer.

    Stronger than field equality on purpose: two selections that merely agree today are two
    things that can drift, and three judges sharing one blindness must not become three judges
    with three slightly different sights. The self-review pass is the Engineer's own rendered
    instructions, so proving the Engineer's block equals the Reviewer's proves all three.
    """
    snapshot = await real_snapshot()
    detail = await real_detail(node_ids=[REAL_FRAME_NODE_ID])
    plan = plan_naming([REAL_FRAME_NODE_ID])
    self_reviewer = StubSelfReviewer(None)
    reviewer_client = ScriptedTextClient([_review_response([])])

    await EngineerAgent(
        prompt_loader=PromptLoader(),
        coding_executor=MockCodingExecutor(
            file_updates={"src/pages/BulkImportApps.js": "export default null;\n"}
        ),
        self_reviewer=self_reviewer,
    ).run(agent_state(tmp_path, [plan, feature_197_contract(), snapshot, detail]))
    await ReviewerAgent(
        prompt_loader=PromptLoader(),
        llm_client=cast("LLMClient", reviewer_client),
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
                plan,
                code_completion_artifact(),
                snapshot,
                detail,
            ],
        )
    )

    engineer_text = rendered_design_text(self_reviewer.calls[0][0], ENGINEER_HEADING)
    reviewer_text = rendered_design_text(reviewer_client.calls[0][0], REVIEWER_HEADING)

    # The raw rendered characters, not a re-serialization of them. Both blocks come from the
    # same renderer over the same selection, so the strings themselves must be equal --
    # comparing `json.dumps(..., sort_keys=True)` of the parsed blocks would call two blocks
    # "byte identical" that differ in order or spacing, which is drift this test exists to see.
    assert engineer_text == reviewer_text
    engineer_block = rendered_design(self_reviewer.calls[0][0], ENGINEER_HEADING)
    assert [item["node_id"] for item in engineer_block["design_nodes"]] == [REAL_FRAME_NODE_ID]


@pytest.mark.asyncio
async def test_without_a_snapshot_every_judges_prompt_is_unchanged(tmp_path: Path) -> None:
    """Safety rule 3, at the three prompts that would otherwise carry an empty block."""
    self_reviewer = StubSelfReviewer(None)
    reviewer_client = ScriptedTextClient([_review_response([])])

    await EngineerAgent(
        prompt_loader=PromptLoader(),
        coding_executor=MockCodingExecutor(
            file_updates={"src/pages/BulkImportApps.js": "export default null;\n"}
        ),
        self_reviewer=self_reviewer,
    ).run(agent_state(tmp_path, [frontend_task_plan(), feature_197_contract()]))
    await ReviewerAgent(
        prompt_loader=PromptLoader(),
        llm_client=cast("LLMClient", reviewer_client),
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

    for prompt in (self_reviewer.calls[0][0], reviewer_client.calls[0][0]):
        assert ENGINEER_HEADING not in prompt
        assert REVIEWER_HEADING not in prompt
        # Nothing from the block: not its heading, not its fields, not its rules. An empty
        # block would be worse than none -- it reads as "there is a design, and it is empty".
        assert "style_name_source" not in prompt
        assert "design_nodes_omitted" not in prompt
        assert "design_nodes_unreachable" not in prompt
        assert "pixel fidelity" not in prompt


def test_a_citation_free_feature_renders_the_pre_coding_prompts_byte_identically() -> None:
    """Safety rule 3 at the product manager and the planner, as byte equality.

    The two templates gate their design block on `design_snapshot | default("")`, and a
    citation-free feature passes the empty string. So the whole rendered prompt must equal,
    byte for byte, the prompt rendered with the variable absent entirely -- which is the
    prompt these templates produced before the design feature existed. Substring checks
    ("no 'Attached design'") pass even when the block leaks whitespace or a stray rule; byte
    equality does not.
    """
    loader = PromptLoader()
    product_manager_context: dict[str, Any] = {"workflow_id": "workflow-1", "prd": "{}"}
    planner_context: dict[str, Any] = {
        "feature_id": "feature-1",
        "technical_prd": "{}",
        "repositories": [],
        "has_reconnaissance": False,
        "reconnaissance": "[]",
    }

    assert loader.render(
        "product_manager/v1.jinja2", **product_manager_context, design_snapshot=""
    ) == loader.render("product_manager/v1.jinja2", **product_manager_context)
    assert loader.render(
        "planner/feature_v1.jinja2", **planner_context, design_snapshot=""
    ) == loader.render("planner/feature_v1.jinja2", **planner_context)


# --------------------------------------------------------------------------------------
# The selection's edges
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_workstream_that_names_no_frame_gets_no_selection() -> None:
    """Every workstream of a feature whose design applies elsewhere, which is most of them."""
    snapshot = await real_snapshot()
    detail = await real_detail()

    assert (
        design_snapshot_context(snapshot=snapshot, detail=detail, task_plan=frontend_task_plan())
        is None
    )
    assert scoped_design_node_ids(frontend_task_plan()) == ()
    assert scoped_design_node_ids(plan_naming([REAL_FRAME_NODE_ID])) == (REAL_FRAME_NODE_ID,)
    # And no snapshot at all is the other `None` case -- a detail artifact without one cannot
    # happen, and would not be a design this workstream is judged against if it did.
    assert (
        design_snapshot_context(
            snapshot=None, detail=detail, task_plan=plan_naming([REAL_FRAME_NODE_ID])
        )
        is None
    )


@pytest.mark.asyncio
async def test_a_frame_detail_could_not_fit_reaches_the_judges_as_the_omission() -> None:
    """Not silence, and not the index record either. Both would be lies of a different kind.

    A workstream assigned two frames where one did not fit its detail bound sees one design and
    one omission entry with its size and its reason. What it must *not* see is the snapshot's
    own record for the frame that did not fit: that is an index record, with no colour, no type
    and no spacing, and an engineer handed one invents all three -- which passes every gate this
    platform has, because no command it runs checks a colour.
    """
    snapshot = await real_snapshot()
    detail = await real_detail(node_ids=[REAL_FRAME_NODE_ID, "12:53"], node_character_bound=8_000)

    context = design_snapshot_context(
        snapshot=snapshot, detail=detail, task_plan=plan_naming([REAL_FRAME_NODE_ID, "12:53"])
    )

    assert context is not None
    assert [item["node_id"] for item in context["design_nodes"]] == ["12:53"]
    assert [item["node_id"] for item in context["design_nodes_omitted"]] == [REAL_FRAME_NODE_ID]
    assert context["design_nodes_omitted"][0]["reason"] == "over_per_node_character_bound"
    assert context["design_nodes_omitted"][0]["characters"] > 8_000
    assert context["design_nodes_not_in_snapshot"] == []
    # The frame that *was* quoted carries the values an engineer builds from.
    assert design_content_is_buildable(context["design_nodes"][0]["design"])
    # And the omitted one is nowhere in the block as a design, only as an omission.
    quoted_ids = {item["node_id"] for item in context["design_nodes"]}
    assert REAL_FRAME_NODE_ID not in quoted_ids


@pytest.mark.asyncio
async def test_the_index_record_is_never_quoted_to_a_builder() -> None:
    """Risk 1 of 96-, asserted rather than trusted.

    The snapshot indexes every cited frame, so for any frame in scope there is always an index
    record available to fall back on. Falling back to it is the tempting wrong answer, and this
    is the test that fixes the right one: a frame the flow has but this workstream was not
    given is reported, with a reason that says no bound hid it -- there was simply nothing
    resolved to quote.
    """
    snapshot = await real_snapshot()

    # No detail at all: a mock composition, a feature whose state predates the two tiers, or a
    # resume that lost the artifact.
    context = design_snapshot_context(
        snapshot=snapshot, detail=None, task_plan=plan_naming([REAL_FRAME_NODE_ID])
    )

    assert context is not None
    assert context["design_nodes"] == [], "an index record reached a builder"
    assert [item["node_id"] for item in context["design_nodes_omitted"]] == [REAL_FRAME_NODE_ID]
    assert context["design_nodes_omitted"][0]["reason"] == "detail_not_resolved_for_this_workstream"
    # Not reported as a frame the design does not have -- it does have it, and saying otherwise
    # would send an engineer to `summary` to report a planning defect that is not there.
    assert context["design_nodes_not_in_snapshot"] == []

    # And every index record in the snapshot really is unbuildable, so the guard above is not
    # vacuous: this is what would have been quoted.
    assert snapshot.nodes
    for record in snapshot.nodes:
        assert not design_content_is_buildable(record.content)


@pytest.mark.asyncio
async def test_a_frame_the_snapshot_never_contained_is_named_rather_than_dropped() -> None:
    """The planner refuses this before a plan is written, so reaching it means a stale plan."""
    snapshot = await real_snapshot()
    detail = await real_detail(node_ids=[REAL_FRAME_NODE_ID])

    context = design_snapshot_context(
        snapshot=snapshot, detail=detail, task_plan=plan_naming([REAL_FRAME_NODE_ID, "999:999"])
    )

    assert context is not None
    assert [item["node_id"] for item in context["design_nodes"]] == [REAL_FRAME_NODE_ID]
    assert context["design_nodes_not_in_snapshot"] == ["999:999"]


@pytest.mark.asyncio
async def test_the_selection_reads_both_artifacts_out_of_state_and_scopes_them() -> None:
    """One entry point, two callers, and the same inputs -- so they cannot disagree."""
    snapshot = await real_snapshot(workflow_id="workflow-1")
    detail = await real_detail(workflow_id="workflow-1", node_ids=[REAL_FRAME_NODE_ID])
    state = {"workflow_id": "workflow-1", "artifacts": [snapshot, detail]}

    context = design_snapshot_context_from_state(
        cast("Any", state), plan_naming([REAL_FRAME_NODE_ID])
    )

    assert context is not None
    assert [item["node_id"] for item in context["design_nodes"]] == [REAL_FRAME_NODE_ID]
    assert design_content_is_buildable(context["design_nodes"][0]["design"])
    # A snapshot belonging to another workflow is not this workflow's design.
    other = {"workflow_id": "workflow-2", "artifacts": [snapshot, detail]}
    assert (
        design_snapshot_context_from_state(cast("Any", other), plan_naming([REAL_FRAME_NODE_ID]))
        is None
    )


@pytest.mark.asyncio
async def test_a_detail_resolved_against_an_older_snapshot_revision_is_not_quoted() -> None:
    """The pairing `snapshot_artifact_id` exists for is checked where it matters.

    A refresh makes a new snapshot revision. A detail resolved against the previous one
    describes a design this attempt is no longer judged against, so quoting it would be quoting
    the wrong design while claiming it is the right one -- and 96-'s whole claim that "attempt 2
    built against the same design attempt 1 did" is checkable rests on this.
    """
    snapshot = await real_snapshot(workflow_id="workflow-1")
    refreshed = snapshot.model_copy(update={"artifact_id": "018_design_snapshot.revision-2.json"})
    stale = await real_detail(
        workflow_id="workflow-1",
        node_ids=[REAL_FRAME_NODE_ID],
        snapshot_artifact_id=snapshot.artifact_id,
    )
    state = {"workflow_id": "workflow-1", "artifacts": [refreshed, stale]}

    assert design_detail_in_state(cast("Any", state)) is None
    context = design_snapshot_context_from_state(
        cast("Any", state), plan_naming([REAL_FRAME_NODE_ID])
    )
    assert context is not None
    assert context["design_nodes"] == []
    assert context["design_nodes_omitted"][0]["reason"] == "detail_not_resolved_for_this_workstream"

    # Paired against the revision it was resolved for, it is quoted.
    paired = {"workflow_id": "workflow-1", "artifacts": [snapshot, stale]}
    assert design_detail_in_state(cast("Any", paired)) is stale


# --------------------------------------------------------------------------------------
# The planner's own validation refuses an invented frame
# --------------------------------------------------------------------------------------


async def plan_with_design_nodes(nodes: list[str]) -> RepositoryExecutionPlanArtifact:
    """A real execution plan whose one workstream is assigned these frames.

    Built by the deterministic planner rather than hand-written, so the plan this validation
    runs against is one the platform actually produces.
    """
    from state.feature_models import RepositorySpec

    _architecture, _contract, plan = await DeterministicFeaturePlanner().plan(
        feature_id="workflow-1",
        technical_prd=technical_prd_artifact(),
        repositories=[
            RepositorySpec(
                repository_id="frontend",
                name="Frontend",
                role="frontend",
                repository_url=cast("Any", "https://github.com/example/frontend"),
                default_branch="main",
            )
        ],
    )
    workstreams = [item.model_copy(update={"design_nodes": nodes}) for item in plan.workstreams]
    return plan.model_copy(update={"workstreams": workstreams})


@pytest.mark.asyncio
async def test_a_planner_assignment_naming_an_unknown_frame_is_refused() -> None:
    """Refused by the planner rather than reaching an engineer.

    A frame id the model invented becomes an instruction to build a screen nobody drew and a
    criterion no diff can satisfy -- the design analogue of an invented contract section, and
    it is checked against the same authority the selection quotes from.
    """
    snapshot = await real_snapshot()

    with pytest.raises(AgentArtifactError, match="must exist in this feature's resolved"):
        _ensure_plan_names_only_real_design_nodes(
            await plan_with_design_nodes(["999:999"]), snapshot
        )
    # A frame the snapshot does contain is accepted.
    _ensure_plan_names_only_real_design_nodes(
        await plan_with_design_nodes([REAL_FRAME_NODE_ID]), snapshot
    )
    # And a plan that assigns nothing never consults the snapshot at all, which is what keeps
    # a feature with no design unaffected.
    _ensure_plan_names_only_real_design_nodes(await plan_with_design_nodes([]), None)


@pytest.mark.asyncio
async def test_an_omitted_frame_is_still_assignable_because_the_design_has_it() -> None:
    """The authority is everything the snapshot mentions, not only what it quoted.

    A frame the resolution could not fit is still a frame the design has: assigning it is
    legitimate, and the engineer is told about the omission rather than refused a plan.
    """
    client = ScriptedFigmaClient(
        nodes={REAL_FILE_KEY: one_file([captured_subtree(), captured_subtree("12:53")])}
    )
    from artifacts.design_references import DesignReference

    # 2,500 sits between the two real frames' measured *index* sizes: in index mode 10:11
    # renders to 3,040 characters and 12:53 to 2,371, so exactly one is omitted and one is
    # indexed. (It was 2,456 and 1,940 before every node started carrying its Figma id, which
    # the asset export addresses image fills by.) The index bound is the one to override here
    # rather than the detail bound, because what the planner is checked against is what the
    # snapshot mentions -- and after 96- the snapshot is the index.
    snapshot = await resolver(client, index_node_character_bound=2_500).resolve(
        feature_id="workflow-1",
        references=[
            DesignReference(
                url=f"https://www.figma.com/design/{REAL_FILE_KEY}/Untitled?node-id=10-11,12-53"
            )
        ],
        artifact_id=FEATURE_ARTIFACT_FILENAMES["design_snapshot"],
    )
    del client

    assert snapshot.design_nodes_omitted, "the bound bit, which is what this case needs"
    omitted = snapshot.design_nodes_omitted[0].node_id
    assert omitted in design_node_names(snapshot)

    _ensure_plan_names_only_real_design_nodes(await plan_with_design_nodes([omitted]), snapshot)


@pytest.mark.asyncio
async def test_a_mock_feature_scopes_its_design_to_the_repositories_the_citation_named() -> None:
    """`applies_to` constrains the assignment; an empty one leaves it to the plan."""
    from agents.planner.feature_planner import _mock_design_nodes
    from artifacts.design_references import DesignReference
    from state.feature_models import RepositorySpec

    scoped = await DeterministicDesignResolver().resolve(
        feature_id="workflow-1",
        references=[
            DesignReference(url=REAL_FRAME_URL, applies_to=["frontend"]),
            DesignReference(
                url=f"https://www.figma.com/design/{REAL_FILE_KEY}/Untitled?node-id=12-53"
            ),
        ],
        artifact_id=FEATURE_ARTIFACT_FILENAMES["design_snapshot"],
    )
    frontend = RepositorySpec(
        repository_id="frontend",
        name="Frontend",
        role="frontend",
        repository_url=cast("Any", "https://github.com/example/frontend"),
        default_branch="main",
    )
    backend = frontend.model_copy(
        update={"repository_id": "backend", "name": "Backend", "role": "backend"}
    )

    # The scoped frame goes only where its author scoped it; the unscoped one goes everywhere,
    # because a mock planner has no basis for choosing and must not read `role`.
    assert _mock_design_nodes(frontend, scoped) == [REAL_FRAME_NODE_ID, "12:53"]
    assert _mock_design_nodes(backend, scoped) == ["12:53"]
    assert _mock_design_nodes(frontend, None) == []
