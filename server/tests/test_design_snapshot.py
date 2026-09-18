"""The citation is resolved once, into an artifact, and the artifact is honest about it.

Three properties this file defends.

**Extraction is checked against a real Figma payload, never a hand-written one.** Both fixtures
under `tests/fixtures/figma/` were captured from live calls on 2026-09-07 -- see that
directory's README for what was removed and why. A hand-written double agrees with the code
reading it by construction, which is how the web client lost a dozen fields, and it would have
hidden every field Figma actually sends that this extraction does not read.

**Every cap says what it dropped.** An unreported cap reads as "the whole design was
considered", so each of the three omission lists is populated independently here.

**Nothing about a citation-free feature changes.** No step, no operation row, no artifact.
"""

from __future__ import annotations

import json
import subprocess
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import TypeAdapter, ValidationError

from adapters.figma_adapter import (
    FigmaClientError,
    FigmaDesignClient,
    FigmaFailureMode,
    FigmaNodesResult,
    FigmaNodeSubtree,
    FigmaTopLevelFrame,
    FigmaTopLevelResult,
)
from agents.shared.contracts import (
    ARTIFACT_FILENAMES,
    FEATURE_ARTIFACT_FILENAMES,
    create_artifact,
)
from api.control_plane import RequestScopedCredentials
from api.feature_control_plane import _initial_feature_state
from api.feature_schemas import StartFeatureRequest
from artifacts.design_references import (
    MAX_DESIGN_DEPTH,
    MAX_DESIGN_DETAIL_DEPTH,
    MAX_DESIGN_DETAIL_INDEX_DEPTH,
    MAX_DESIGN_DETAIL_NODE_CHARACTERS,
    MAX_DESIGN_DETAIL_TOTAL_CHARACTERS,
    MAX_DESIGN_INDEX_NODE_CHARACTERS,
    MAX_DESIGN_NODES,
    DesignReference,
)
from artifacts.schemas import (
    Artifact,
    DesignDetailArtifact,
    DesignSnapshotArtifact,
    PRDArtifact,
    RepositoryWorkstreamPlan,
    TechnicalPRDArtifact,
)
from services.cancellation import MockCancellationToken
from services.design_resolution import (
    _MIN_DETAIL_DEPTH,
    DESIGN_BUILD_ONLY_KEYS,
    DESIGN_INDEX_KEYS,
    STYLE_NAMES_FROM_NODES,
    DesignReferenceResolver,
    DesignResolutionRefused,
    DeterministicDesignResolver,
    FigmaDesignResolver,
    _characters,
    design_content_is_buildable,
    design_detail_artifact_id,
    design_snapshot_artifact_id,
    extract_design_node,
    unreachable_design_detail,
)
from state.external_operations import ExternalOperationType
from state.feature_models import RepositorySpec
from tests.test_design_source import app_with_design_source as _app_with_design_source
from tests.test_design_source import client as _client
from tests.test_design_source import operator_headers as _operator_headers
from tests.test_feature_api import feature_payload
from workflows.feature_workflow import (
    FeatureStep,
    FeatureWorkflowOrchestrator,
    is_transient_provider_fault,
    next_step,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "figma"

# The real file these fixtures were captured from, and a real frame inside it.
REAL_FILE_KEY = "VGULlnz44R0Ooe4FZKDxlhh4"
REAL_FRAME_NODE_ID = "10:11"
REAL_FRAME_NAME = "#FClock"

# The second real file, captured 2026-09-10: a product archive with a design system inside it.
# `PRODUCT_FRAME_NODE_ID` is the frame every number in 96- is calibrated on;
# `PRODUCT_FORM_NODE_ID` is a self-contained sign-in form inside it, at a natural depth of 5,
# which is the one real screen measured that fits today's bounds without truncation.
PRODUCT_FILE_KEY = "EqhdoZR1UKVZsaX7jplDsP"
PRODUCT_FRAME_NODE_ID = "73996:19746"
PRODUCT_FORM_NODE_ID = "74046:28253"


def captured_file() -> dict[str, Any]:
    """Load the live-captured file read, exactly as Figma answered it."""
    payload: dict[str, Any] = json.loads((FIXTURES / "figma_file_read.json").read_text())
    return payload


def captured_subtree(node_id: str = REAL_FRAME_NODE_ID) -> FigmaNodeSubtree:
    """Build the subtree the nodes endpoint returns, from the captured file read.

    The file endpoint returns the identical node shape and the identical `styles`,
    `components` and `componentSets` maps one level up -- verified against both endpoints on
    the same file. So this is the real payload, addressed the way the adapter addresses it,
    rather than a second shape invented to suit the test.
    """
    payload = captured_file()
    for canvas in payload["document"]["children"]:
        for frame in canvas.get("children") or []:
            if frame["id"] == node_id:
                return FigmaNodeSubtree(
                    node_id=node_id,
                    document=frame,
                    styles=payload.get("styles") or {},
                    components=payload.get("components") or {},
                    component_sets=payload.get("componentSets") or {},
                )
    msg = f"the captured payload has no node {node_id}"
    raise AssertionError(msg)


def product_subtree(node_id: str = PRODUCT_FRAME_NODE_ID) -> FigmaNodeSubtree:
    """Build a subtree from the captured product frame, root or any node inside it.

    A second real file, and a very different one: `figma_file_read.json` uses no auto layout
    and defines no named styles, so it cannot measure either. This one is a product archive
    with a design system inside it -- 83 of its measured frame's 93 nodes are auto-layout
    containers, all 16 text nodes carry typography, and its type styles are named
    (`Label/Label-3/Medium`). It is the payload 96-'s bounds are calibrated on, which is why
    the bound tests read it rather than a synthetic frame sized to suit them.
    """
    payload: dict[str, Any] = json.loads((FIXTURES / "figma_product_frame.json").read_text())
    entry = payload["nodes"][PRODUCT_FRAME_NODE_ID]
    maps: dict[str, Any] = {
        "styles": entry.get("styles") or {},
        "components": entry.get("components") or {},
        "component_sets": entry.get("componentSets") or {},
    }

    def find(node: dict[str, Any]) -> dict[str, Any] | None:
        if node.get("id") == node_id:
            return node
        for child in node.get("children") or []:
            if isinstance(child, dict) and (found := find(child)) is not None:
                return found
        return None

    document = find(entry["document"])
    if document is None:
        msg = f"the captured product frame has no node {node_id}"
        raise AssertionError(msg)
    return FigmaNodeSubtree(node_id=node_id, document=document, **maps)


def citation(node_ids: list[str] | None = None, **overrides: Any) -> DesignReference:
    """One citation against the real file, in the URL form a browser produces."""
    query = "" if node_ids is None else "?node-id=" + ",".join(node_ids)
    return DesignReference(
        url=f"https://www.figma.com/design/{REAL_FILE_KEY}/Untitled{query}",
        **overrides,
    )


class _ScriptedFigmaClient(FigmaDesignClient):
    """Answer with prepared results, and record exactly what was asked for."""

    def __init__(
        self,
        *,
        nodes: dict[str, FigmaNodesResult] | None = None,
        top_level: FigmaTopLevelResult | None = None,
        raises: FigmaClientError | None = None,
        rendered: bytes | None = None,
        render_raises: FigmaClientError | None = None,
    ) -> None:
        """Bind what this client will answer, per file key."""
        self._nodes = nodes or {}
        self._top_level = top_level
        self._raises = raises
        self._rendered = rendered
        self._render_raises = render_raises
        self.node_calls: list[tuple[str, list[str]]] = []
        self.top_level_calls: list[str] = []
        self.render_calls: list[tuple[str, str]] = []

    async def fetch_nodes(self, file_key: str, node_ids: Any) -> FigmaNodesResult:
        """Record the call and answer, or raise what this client was told to raise."""
        self.node_calls.append((file_key, list(node_ids)))
        if self._raises is not None:
            raise self._raises
        return self._nodes[file_key]

    async def fetch_top_level_frames(self, file_key: str, *, limit: int) -> FigmaTopLevelResult:
        """Record the call and answer a whole-file citation."""
        self.top_level_calls.append(file_key)
        if self._raises is not None:
            raise self._raises
        assert self._top_level is not None
        return self._top_level

    async def render_preview(self, file_key: str, node_id: str, *, version: str) -> str:
        """Record the render and answer a URL, or raise what this client was told to raise."""
        del version
        self.render_calls.append((file_key, node_id))
        if self._render_raises is not None:
            raise self._render_raises
        return f"https://figma-alpha-api.s3.us-west-2.amazonaws.com/{node_id}.png"

    async def fetch_rendered_bytes(self, url: str) -> bytes:
        """Answer the prepared bytes for a render this client agreed to make."""
        del url
        if self._render_raises is not None:
            raise self._render_raises
        return self._rendered or b""


def resolver(client: FigmaDesignClient, **bounds: Any) -> FigmaDesignResolver:
    """A resolver over one scripted client, with the shipped bounds unless overridden."""

    async def factory() -> FigmaDesignClient | None:
        return client

    return FigmaDesignResolver(client_factory=factory, **bounds)


def one_file(subtrees: list[FigmaNodeSubtree], absent: tuple[str, ...] = ()) -> FigmaNodesResult:
    """One nodes answer for the real file, carrying the version it reported."""
    return FigmaNodesResult(
        file_key=REAL_FILE_KEY,
        file_name="Untitled",
        file_version=str(captured_file()["version"]),
        subtrees=tuple(subtrees),
        absent_node_ids=absent,
    )


# --------------------------------------------------------------------------------------
# Extraction, against the payload Figma actually sent
# --------------------------------------------------------------------------------------


def test_extraction_reads_a_real_captured_frame() -> None:
    """The frame, its text, its type and its component names -- from the live payload.

    Read out of a genuine capture rather than a double, so a field this repository invented or
    a field Figma sends under another name fails here rather than passing against a fixture
    that was written to agree.
    """
    content, rendered, depth = extract_design_node(captured_subtree())

    assert content["name"] == REAL_FRAME_NAME
    assert content["type"] == "FRAME"
    assert content["path"] == REAL_FRAME_NAME
    # A real measurement, not a round number: 1204x643 is what this frame is.
    assert content["size"] == {"width": 1204.0, "height": 643.0}
    assert content["constraints"] == {"vertical": "TOP", "horizontal": "LEFT"}

    def walk(node: dict[str, Any]) -> list[dict[str, Any]]:
        found = [node]
        for child in node.get("children") or []:
            found.extend(walk(child))
        return found

    every = walk(content)
    assert len(every) == rendered
    assert depth >= 3, "a real frame is nested, and the extraction follows it"
    # Paths are built from names in document order, so a reader can find a node in Figma.
    assert all(item["path"].startswith(REAL_FRAME_NAME) for item in every[1:])
    # The actual characters, never a summary: "the text is the text" is one of the few design
    # criteria a reviewer reading a diff can check.
    texts = [item for item in every if item["type"] == "TEXT"]
    assert texts, "the captured frame has text in it"
    assert all("text" in item for item in texts)
    assert any(item["text"] == "10:10" for item in texts)
    typography = texts[0]["typography"]
    assert typography["family"] == "Roboto"
    assert typography["weight"] == 700
    assert isinstance(typography["size"], float)
    # Fills reduced to the hex an engineer would otherwise compute, from real 0-1 channels.
    filled = [item for item in every if item.get("fills")]
    assert filled
    assert all(
        entry["hex"].startswith("#") and len(entry["hex"]) == 7
        for item in filled
        for entry in item["fills"]
        if entry["type"] == "SOLID"
    )
    # An instance names the component it came from, which is what says "reuse this".
    instances = [item for item in every if item["type"] == "INSTANCE"]
    assert instances
    assert any(item.get("component", {}).get("name") for item in instances)


# --------------------------------------------------------------------------------------
# Two renderings, two purposes: the index tier (96- Part A)
# --------------------------------------------------------------------------------------


def _every_key(content: dict[str, Any]) -> set[str]:
    """Every key appearing anywhere in a rendered tree, at any depth."""
    keys = set(content)
    for child in content.get("children") or []:
        if isinstance(child, dict):
            keys |= _every_key(child)
    return keys


def _every_value(content: dict[str, Any], key: str) -> list[Any]:
    """Every value one key takes anywhere in a rendered tree, in document order."""
    found = [content[key]] if key in content else []
    for child in content.get("children") or []:
        if isinstance(child, dict):
            found.extend(_every_value(child, key))
    return found


def test_index_mode_renders_the_same_tree_without_any_of_its_values() -> None:
    """The index answers "which frame is this"; build answers "how do I make it".

    Asserted on the real product frame every 96- number is calibrated on, because the whole
    justification for a second rendering is a size measurement, and a size measurement against
    a synthetic frame measures whatever the fixture author chose.

    The keys are the assertion and not the bytes: an index record carrying `fills` at depth
    nine is an index record an engineer could half-build from, which is the failure mode the
    two tiers exist to keep apart.
    """
    subtree = product_subtree()
    build, build_nodes, build_depth = extract_design_node(subtree)
    index, index_nodes, index_depth = extract_design_node(subtree, detail="index")

    # Same tree, walked the same way -- only the fields differ.
    assert (index_nodes, index_depth) == (build_nodes, build_depth) == (93, 8)

    assert _every_key(index) <= DESIGN_INDEX_KEYS
    assert not _every_key(index) & DESIGN_BUILD_ONLY_KEYS
    for absent in ("fills", "typography", "layout", "size", "strokes", "constraints"):
        assert absent not in _every_key(index), f"index mode still carries {absent}"

    # And build mode is unchanged: the mode parameter defaults to what this always produced.
    for present in ("fills", "typography", "layout", "size", "constraints"):
        assert present in _every_key(build)


def test_the_index_keeps_every_name_and_every_character_build_fidelity_carries() -> None:
    """Nothing a role *deciding* needs is lost, which is what makes the index honest.

    Names outrank values is this module's rule, and the index is that rule taken to its end:
    what it drops is exactly the values. So every `text`, `style_names` and `component` build
    fidelity resolves must still be there, in the same order -- a planner assigning frames
    reads the names, and a product manager deriving requirements reads the copy.
    """
    subtree = product_subtree()
    build, _, _ = extract_design_node(subtree)
    index, _, _ = extract_design_node(subtree, detail="index")

    for key in ("text", "style_names", "component", "variant_properties", "path", "name", "type"):
        assert _every_value(index, key) == _every_value(build, key), f"the index lost {key}"

    # Not vacuous: this frame really does carry all four, which is why it is the fixture.
    assert len(_every_value(index, "text")) == 16
    assert len(_every_value(index, "style_names")) == 16
    assert len(_every_value(index, "component")) == 13
    assert "Label/Label-3/Medium" in json.dumps(index)


def test_the_index_is_two_thirds_smaller_and_the_measurement_is_recorded_here() -> None:
    """The size claim the whole two-tier design rests on, measured rather than asserted.

    Recorded as exact numbers the way `test_no_bound_bites_on_the_real_file_that_calibrated_them`
    records its own, so a change to `_render` that quietly re-inflates the index shows up here
    as a drift rather than as a slow return to one tier.

    Measured through `_characters`, which is `indent=2` -- the same serialization
    `design_snapshot_context_json` hands a model. 96- was first written against a compact
    measurement and every bound it proposed was consequently 2.6x too small; the numbers below
    are the indented ones, which are the ones the bounds are compared against.
    """
    subtree = product_subtree()
    build, nodes, _ = extract_design_node(subtree)
    index, _, _ = extract_design_node(subtree, detail="index")

    build_characters = len(json.dumps(build, indent=2, sort_keys=True, ensure_ascii=False))
    index_characters = len(json.dumps(index, indent=2, sort_keys=True, ensure_ascii=False))

    assert (build_characters, index_characters) == (142_082, 47_200)
    assert nodes == 93
    assert index_characters * 2 < build_characters, "the index must be at least 50% smaller"
    # ~507 a node now that each carries its Figma id; it was ~450 before the asset
    # export needed a way to address an image fill.
    assert index_characters // nodes == 507

    # And the index bound is headroom on that rather than a trim on it -- the rule every bound
    # in `design_references.py` is written to. A first draft of 96- proposed 30,000 here, which
    # this frame alone would have exceeded.
    assert index_characters < MAX_DESIGN_INDEX_NODE_CHARACTERS
    assert index_characters * 3 < MAX_DESIGN_INDEX_NODE_CHARACTERS


def test_an_index_record_is_never_mistaken_for_something_an_engineer_can_build() -> None:
    """Risk 1 of 96-, guarded by a predicate rather than by remembering.

    A valueless rendering reaching the Engineer produces invented hex codes, and invented hex
    codes pass every gate this platform has, because no command it runs checks a colour. So
    "is this buildable" is a question with an answer in code, asked of the whole tree: a frame
    whose root declares nothing but whose children carry the paint is buildable, and a frame
    with no value anywhere is not, whatever produced it.
    """
    subtree = product_subtree()
    build, _, _ = extract_design_node(subtree)
    index, _, _ = extract_design_node(subtree, detail="index")

    assert design_content_is_buildable(build)
    assert not design_content_is_buildable(index)

    # Asked of the tree and not the root: strip every build-only key from the root alone and
    # the record is still buildable, because its children still say what to build.
    root_stripped = {
        key: value for key, value in build.items() if key not in DESIGN_BUILD_ONLY_KEYS
    }
    assert design_content_is_buildable(root_stripped)

    # The deterministic resolver's own content must pass, or every mock-tier feature would be
    # reported as having been handed an unbuildable design.
    assert design_content_is_buildable(
        {"path": "mock/x", "name": "x", "type": "FRAME", "layout": {"layoutMode": "VERTICAL"}}
    )


# What `MAX_DESIGN_NODE_CHARACTERS` was before 96- raised and renamed it. Kept as a literal
# because the frame below had to fit *that* number for the design path's first live run to be
# possible at all -- a claim which stops meaning anything if it is checked against the bound
# this item shipped, which is more than ten times larger.
_PRE_96_NODE_CHARACTER_BOUND = 40_000


def test_the_form_that_fits_every_bound_is_what_the_first_live_run_cited() -> None:
    """The one real screen measured that resolves whole, at its natural depth.

    Recorded here because 96-'s prerequisite run depended on finding it: every top-level frame
    on this file is over the *old* per-node bound, so a live exercise of the design path needed
    a frame that fits it. This is that frame -- a sign-in form, self-contained at depth 5, so
    nothing about it is truncated, omitted or reported in either tier. If a bound change makes
    this frame stop fitting, the design path has no live smoke test left.
    """
    content, nodes, depth = extract_design_node(product_subtree(PRODUCT_FORM_NODE_ID))
    characters = len(json.dumps(content, indent=2, sort_keys=True, ensure_ascii=False))

    assert (characters, nodes, depth) == (37_139, 36, 5)
    assert depth < MAX_DESIGN_DEPTH, "nothing about this frame is cut by either depth bound"
    assert characters < _PRE_96_NODE_CHARACTER_BOUND, "it resolved before this item, too"
    assert characters < MAX_DESIGN_DETAIL_NODE_CHARACTERS
    # Real design-system content, which is what makes it worth citing: named type styles, the
    # component sets they come from, and the literal copy on the screen.
    rendered = json.dumps(content)
    assert "Label/Label-3/Medium" in rendered
    assert "Text Field" in rendered
    assert "Button Contained" in rendered
    assert "Forgot password" in _every_value(content, "text")
    assert "Sign in" in _every_value(content, "text")


def test_a_depth_bound_reports_the_children_it_did_not_follow() -> None:
    """A depth cut that silently dropped children would read as a frame that has none.

    Which is exactly the misreading the omission lists exist to prevent, so the cut is
    reported on the node it happened at.
    """
    subtree = captured_subtree()
    deep, _, natural_depth = extract_design_node(subtree)
    shallow, _, cut_depth = extract_design_node(subtree, depth_bound=1)

    assert natural_depth > 1
    assert cut_depth == 1
    assert "children_beyond_depth_bound" in json.dumps(shallow)
    assert len(json.dumps(shallow)) < len(json.dumps(deep))
    # The bound at the frame's own natural depth truncates nothing.
    whole, _, _ = extract_design_node(subtree, depth_bound=natural_depth)
    assert "children_beyond_depth_bound" not in json.dumps(whole)


def test_style_names_come_before_the_values_they_resolve_to() -> None:
    """Names outrank values, and the rendering says so by putting them first.

    A hex code tells an engineer what to hardcode; `color/surface/raised` tells it what the
    repository already has. The captured file defines no named styles, so this asserts the
    mechanism against a style map, using the same real node.
    """
    payload = captured_file()
    frame = next(
        frame
        for canvas in payload["document"]["children"]
        for frame in (canvas.get("children") or [])
        if frame["id"] == REAL_FRAME_NODE_ID
    )
    # The `styles` reference and the `styles` map are both Figma's own shapes, and this is the
    # pairing the live API returns for a file that uses named styles -- measured on
    # `28gd2JrZO28FCN9PCKM4qK`, whose login frame returns `{"fills": "309:7176"}` against a map
    # entry named `Primary / 100`.
    named = FigmaNodeSubtree(
        node_id=REAL_FRAME_NODE_ID,
        document={**frame, "styles": {"fills": "309:7176"}},
        styles={"309:7176": {"key": "k", "name": "Primary / 100", "styleType": "FILL"}},
    )

    content, _, _ = extract_design_node(named)

    assert content["style_names"] == {"fills": "Primary / 100"}
    keys = list(content)
    assert keys.index("style_names") < keys.index("fills"), (
        "the name has to be read before the value it resolves to"
    )


# --------------------------------------------------------------------------------------
# The three omission lists, each populated on its own
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_frame_over_the_per_node_bound_is_omitted_whole_and_named() -> None:
    """Never trimmed. A judge shown two thirds of a frame finds a violation that is not there."""
    subtree = captured_subtree()
    client = _ScriptedFigmaClient(nodes={REAL_FILE_KEY: one_file([subtree])})

    with pytest.raises(DesignResolutionRefused):
        # With the bound below the frame's real size, nothing resolves -- and a feature with a
        # mock attached and no snapshot must stop rather than be built against prose.
        await resolver(client, index_node_character_bound=100).resolve(
            feature_id="feature-1",
            references=[citation([REAL_FRAME_NODE_ID], label="the clock")],
            artifact_id=FEATURE_ARTIFACT_FILENAMES["design_snapshot"],
        )

    # With a second frame that does fit, the oversized one costs only itself.
    small = captured_subtree("14:0")
    client = _ScriptedFigmaClient(nodes={REAL_FILE_KEY: one_file([subtree, small])})
    snapshot = await resolver(client, index_node_character_bound=2_000).resolve(
        feature_id="feature-1",
        references=[citation([REAL_FRAME_NODE_ID, "14:0"], label="two frames")],
        artifact_id=FEATURE_ARTIFACT_FILENAMES["design_snapshot"],
    )

    assert [item.node_id for item in snapshot.nodes] == ["14:0"]
    assert len(snapshot.design_nodes_omitted) == 1
    omission = snapshot.design_nodes_omitted[0]
    assert omission.node_id == REAL_FRAME_NODE_ID
    assert omission.reason == "over_index_per_node_character_bound"
    assert omission.characters > 2_000, "the size it measured is recorded, not just the bound"
    assert omission.label == "two frames"


@pytest.mark.asyncio
async def test_the_total_bound_reports_what_it_dropped_and_keeps_what_fit() -> None:
    """The bound is spent in citation order, and a frame that does not fit costs only itself."""
    first = captured_subtree(REAL_FRAME_NODE_ID)
    second = captured_subtree("12:53")
    client = _ScriptedFigmaClient(nodes={REAL_FILE_KEY: one_file([first, second])})

    snapshot = await resolver(client, index_total_character_bound=3_500).resolve(
        feature_id="feature-1",
        references=[citation([REAL_FRAME_NODE_ID, "12:53"])],
        artifact_id=FEATURE_ARTIFACT_FILENAMES["design_snapshot"],
    )

    assert [item.node_id for item in snapshot.nodes] == [REAL_FRAME_NODE_ID]
    assert [item.reason for item in snapshot.design_nodes_omitted] == ["index_total_exhausted"]
    assert snapshot.characters_selected == snapshot.nodes[0].characters
    assert snapshot.bounds["characters_total"] == 3_500


@pytest.mark.asyncio
async def test_the_node_count_bound_reports_what_it_dropped() -> None:
    """A separate list entry with its own reason, so "why is this frame missing" has an answer."""
    subtrees = [captured_subtree(node_id) for node_id in (REAL_FRAME_NODE_ID, "14:0", "12:53")]
    client = _ScriptedFigmaClient(nodes={REAL_FILE_KEY: one_file(subtrees)})

    snapshot = await resolver(client, node_bound=2).resolve(
        feature_id="feature-1",
        references=[citation([REAL_FRAME_NODE_ID, "14:0", "12:53"])],
        artifact_id=FEATURE_ARTIFACT_FILENAMES["design_snapshot"],
    )

    assert len(snapshot.nodes) == 2
    assert [item.reason for item in snapshot.design_nodes_omitted] == ["over_node_count_bound"]
    assert snapshot.design_nodes_omitted[0].node_id == "12:53"


@pytest.mark.asyncio
async def test_a_frame_the_file_does_not_define_is_recorded_as_absent() -> None:
    """Figma answers `200` with `nodes[id] = null`, which is an answer and not a failure.

    Read straight from the live capture of exactly that case, so this is the real behaviour
    rather than an assumption about it.
    """
    captured = json.loads((FIXTURES / "figma_nodes_absent.json").read_text())
    assert captured["nodes"] == {"1:2": None}, "the captured payload is that exact answer"

    client = _ScriptedFigmaClient(
        nodes={REAL_FILE_KEY: one_file([captured_subtree()], absent=("1:2",))}
    )
    snapshot = await resolver(client).resolve(
        feature_id="feature-1",
        references=[citation([REAL_FRAME_NODE_ID, "1:2"], label="deleted?")],
        artifact_id=FEATURE_ARTIFACT_FILENAMES["design_snapshot"],
    )

    assert [item.node_id for item in snapshot.nodes] == [REAL_FRAME_NODE_ID]
    assert len(snapshot.design_nodes_absent) == 1
    assert snapshot.design_nodes_absent[0].node_id == "1:2"
    assert snapshot.design_nodes_absent[0].reason == "not_defined_in_file"
    assert snapshot.design_nodes_unreachable == []
    assert snapshot.design_nodes_omitted == []


@pytest.mark.asyncio
async def test_a_file_the_token_cannot_read_is_recorded_as_unreachable() -> None:
    """A 404 -- which is what Figma answers for both "gone" and "you cannot see it"."""
    refusal = FigmaClientError(
        "refused",
        mode=FigmaFailureMode.FILE_UNREACHABLE,
        error_code="figma_file_unreachable_404",
        endpoint="files.nodes",
        provider_status=404,
    )
    client = _ScriptedFigmaClient(raises=refusal)

    with pytest.raises(DesignResolutionRefused) as refused:
        await resolver(client).resolve(
            feature_id="feature-1",
            references=[citation([REAL_FRAME_NODE_ID])],
            artifact_id=FEATURE_ARTIFACT_FILENAMES["design_snapshot"],
        )

    # Nothing resolved, so the feature stops before planning with something actionable.
    diagnostic = refused.value.diagnostics[0]
    assert REAL_FILE_KEY in diagnostic
    assert "could not read" in diagnostic
    assert "Settings" in diagnostic


@pytest.mark.asyncio
async def test_one_unreachable_file_does_not_discard_the_frames_that_did_resolve() -> None:
    """A snapshot holding two of three frames is useful, and the third is named.

    The partial case is deliberately not a stop: the engineer is shown what it was not shown,
    which is the whole point of the omission lists.
    """
    other_key = "28gd2JrZO28FCN9PCKM4qK"

    class _MixedClient(_ScriptedFigmaClient):
        async def fetch_nodes(self, file_key: str, node_ids: Any) -> FigmaNodesResult:
            self.node_calls.append((file_key, list(node_ids)))
            if file_key == other_key:
                raise FigmaClientError(
                    "refused",
                    mode=FigmaFailureMode.FILE_UNREACHABLE,
                    error_code="figma_file_unreachable_404",
                    endpoint="files.nodes",
                    provider_status=404,
                )
            return one_file([captured_subtree()])

    client = _MixedClient()
    snapshot = await resolver(client).resolve(
        feature_id="feature-1",
        references=[
            citation([REAL_FRAME_NODE_ID], label="readable"),
            DesignReference(
                url=f"https://www.figma.com/design/{other_key}/Other?node-id=2-303",
                label="not readable",
            ),
        ],
        artifact_id=FEATURE_ARTIFACT_FILENAMES["design_snapshot"],
    )

    assert [item.node_id for item in snapshot.nodes] == [REAL_FRAME_NODE_ID]
    assert len(snapshot.design_nodes_unreachable) == 1
    assert snapshot.design_nodes_unreachable[0].file_key == other_key
    assert snapshot.design_nodes_unreachable[0].label == "not readable"


# --------------------------------------------------------------------------------------
# The failure policy: what the provider answered, versus what it did not answer
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "mode", "retryable"),
    [
        # The provider did not answer. Retryable within the operation's own budget.
        (429, FigmaFailureMode.TRANSPORT, True),
        (408, FigmaFailureMode.TRANSPORT, True),
        (500, FigmaFailureMode.TRANSPORT, True),
        (503, FigmaFailureMode.TRANSPORT, True),
        (None, FigmaFailureMode.TRANSPORT, True),
        # The provider answered no. Terminal for this submission: it answers the same way
        # every time, so spending retries on it converts a diagnosable stop into "the
        # provider did not answer".
        (401, FigmaFailureMode.CREDENTIAL_REFUSED, False),
        (403, FigmaFailureMode.CREDENTIAL_REFUSED, False),
        (404, FigmaFailureMode.FILE_UNREACHABLE, False),
        (400, FigmaFailureMode.REQUEST_REFUSED, False),
    ],
)
def test_the_retryable_split_is_read_from_the_status_and_nowhere_else(
    status: int | None, mode: FigmaFailureMode, retryable: bool
) -> None:
    """One predicate, and the workflow reads the adapter's rather than re-deriving it.

    Every status here was observed against the live API on 2026-09-07 except the 5xx family:
    200 for a good token, 401 for a missing or invalid one, 403 for a scope the token does not
    hold, 404 for a file it cannot read, 400 for a malformed node id, and 429 -- repeatedly --
    from the rate limiter.
    """
    error = FigmaClientError(
        "refused",
        mode=mode,
        error_code="figma_test",
        endpoint="files.nodes",
        provider_status=status,
    )

    assert error.retryable is retryable
    # And the workflow's single seam agrees, including through a wrapping cause chain.
    assert is_transient_provider_fault(error) is retryable
    wrapped = RuntimeError("wrapped")
    wrapped.__cause__ = error
    assert is_transient_provider_fault(wrapped) is retryable


@pytest.mark.parametrize(
    ("status", "headers", "expected"),
    [
        # The live 429 that ended AB-Feature-228, verbatim: about seventy-two hours.
        (429, {"retry-after": "261044"}, 261_044),
        (503, {"retry-after": "30"}, 30),
        # Nothing usable, and nothing invented in its place. The HTTP-date form is legal and
        # deliberately unread -- honouring it needs a clock, and a wrong deadline in a record
        # a person acts on is worse than no deadline at all.
        (429, {"retry-after": "Wed, 15 Sep 2026 03:46:00 GMT"}, None),
        (429, {"retry-after": "-5"}, None),
        (429, {"retry-after": "999999999"}, None),
        (429, {}, None),
        # A status the retry window means nothing for never carries one: no retry is made
        # against a 404, so a wait would be describing something that changes nothing.
        (404, {"retry-after": "261044"}, None),
    ],
)
def test_the_wait_a_design_source_asks_for_is_read_as_a_number(
    status: int, headers: dict[str, str], expected: int | None
) -> None:
    """`Retry-After` is an instruction, and it is the difference between weather and a lockout.

    AB-Feature-228 spent all nine of its allowed calls in fifty-one seconds against a 429 whose
    `Retry-After` was three days, and nothing anywhere read the header -- so the record could
    not say, and the operator could not learn, that no resume would work before Tuesday.
    """
    from adapters.figma_adapter import _classify

    class _Response:
        def __init__(self) -> None:
            self.status_code = status
            self.headers = headers

    with pytest.raises(FigmaClientError) as raised:
        _classify("files.nodes", _Response())
    assert raised.value.retry_after_seconds == expected


def test_a_design_fetch_failure_says_which_refusal_it_was_in_the_journal() -> None:
    """The journal row's `error_code` names the failure, not merely that there was one.

    `ExternalOperationExecutor` reads `operation_outcome` off the error, so this is the
    existing seam rather than a new one. Without it AB-Feature-228's nine rows all read
    `fetch_design_reference_failed` -- indistinguishable from a revoked token or an unreadable
    file, which is why answering "why did the design fail" needed a direct call to Figma.
    """
    error = FigmaClientError(
        "refused",
        mode=FigmaFailureMode.TRANSPORT,
        error_code="figma_transport_429",
        endpoint="files.nodes",
        provider_status=429,
    )

    assert error.operation_outcome == "figma_transport_429"


def test_a_refusal_the_person_must_fix_is_never_retried() -> None:
    """`DesignResolutionRefused` is not an infrastructure fault, so it propagates at once.

    Spending a fault allowance on a wrong URL is how AB-Feature-172 spent nine calls proving
    that a deterministic 400 answers identically every time.
    """
    assert is_transient_provider_fault(DesignResolutionRefused("no")) is False


@pytest.mark.asyncio
async def test_a_refused_resolution_is_filed_as_the_persons_to_fix_not_a_platform_defect(
    tmp_path: Path,
) -> None:
    """End to end through the queue: the refusal's classification survives to the record.

    The exception used to declare no classification, so the generic terminal handler read its
    type name, `normalize_classification` found no transient marker in it, and the record said
    `platform_defect` -- sending somebody to debug this codebase over a URL only its author
    can correct. The classification it declares now is the one the platform already uses for
    a refusal a person can fix, and the actionable sentence reaches the summary.
    """
    from state.enums import FeatureWorkflowStatus
    from state.failure_diagnosis import FeatureFailureClassification
    from storage.db import Database
    from storage.feature_store import SqlAlchemyFeatureControlPlane
    from tests.support import drain_feature_queue

    async def no_client() -> FigmaDesignClient | None:
        return None

    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'design-refusal.db'}")
    await database.create_schema()
    store = SqlAlchemyFeatureControlPlane(
        database=database,
        mock_runner=FeatureWorkflowOrchestrator(
            design_resolver=FigmaDesignResolver(client_factory=no_client)
        ),
    )
    payload = feature_payload()
    payload["feature_id"] = "feature-design-refused"
    payload["prd"] = {
        **cast("dict[str, Any]", payload["prd"]),
        "design_references": [
            {"url": f"https://www.figma.com/design/{REAL_FILE_KEY}/Untitled?node-id=10-11"}
        ],
    }
    await store.start(
        StartFeatureRequest.model_validate(payload),
        idempotency_key="design-refused-key",
        credentials=RequestScopedCredentials(openai_api_key=None, github_token=None),
        owner_id="platform-admin",
    )
    await drain_feature_queue(store)

    record = await store.get_record("feature-design-refused")
    assert record.state.status is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
    summary = record.state.failure_summary
    assert summary is not None
    assert summary.root_classification == FeatureFailureClassification.FEATURE_QUEUE_REFUSED.value
    # Not retryable: the provider answered, and answers the same way every time. What moves
    # this feature is a person's correction, which is what the status already says.
    assert summary.retryable is False
    # The sentence the raise site composed, reaching the person who has to act on it.
    assert any("Store one in Settings and resume" in item for item in summary.diagnostics)


# --------------------------------------------------------------------------------------
# Resolved once. A refresh is a revision, never an edit.
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_two_resolutions_produce_two_revisions_and_never_mutate_the_first() -> None:
    """Safety rule 4, in the artifact id.

    Attempt 2 of a workstream builds against the same snapshot attempt 1 did, and a review
    judges against the snapshot the attempt was given -- so the first has to stay readable
    forever. A design edited mid-feature would otherwise silently change the work order and
    leave a diff nobody can explain.
    """
    client = _ScriptedFigmaClient(nodes={REAL_FILE_KEY: one_file([captured_subtree()])})
    resolve = resolver(client)
    reference = citation([REAL_FRAME_NODE_ID], label="first")

    first = await resolve.resolve(
        feature_id="feature-1",
        references=[reference],
        artifact_id=design_snapshot_artifact_id(0),
    )
    second = await resolve.resolve(
        feature_id="feature-1",
        references=[reference],
        artifact_id=design_snapshot_artifact_id(1),
    )

    assert first.artifact_id == "018_design_snapshot.json"
    assert second.artifact_id == "018_design_snapshot.revision-2.json"
    assert design_snapshot_artifact_id(2) == "018_design_snapshot.revision-3.json"
    # The first is untouched, which is what "never an edit of one" means.
    assert first.nodes[0].label == "first"
    assert first.artifact_id != second.artifact_id


@pytest.mark.asyncio
async def test_the_snapshot_records_its_provenance_and_where_its_names_came_from() -> None:
    """The file version pins the console's preview; the name source stops an assumption.

    Figma's images endpoint renders the *current* file unless told otherwise, so the version
    the snapshot recorded is what a later preview is pinned to. And the variables API is out of
    reach for a read-only token -- measured live -- so the snapshot says the names came from
    the nodes rather than leaving a reader to assume the richer source.
    """
    client = _ScriptedFigmaClient(nodes={REAL_FILE_KEY: one_file([captured_subtree()])})

    snapshot = await resolver(client).resolve(
        feature_id="feature-1",
        references=[citation([REAL_FRAME_NODE_ID])],
        artifact_id=FEATURE_ARTIFACT_FILENAMES["design_snapshot"],
    )

    assert [item.file_key for item in snapshot.files] == [REAL_FILE_KEY]
    assert snapshot.files[0].file_version == str(captured_file()["version"])
    assert snapshot.style_name_source == STYLE_NAMES_FROM_NODES
    assert snapshot.nodes[0].source_url.startswith("https://www.figma.com/design/")
    assert snapshot.resolved_at is not None
    assert snapshot.bounds["depth"] > 0
    # And it survives the round trip a persisted artifact takes -- through
    # `model_validate_json`, which is the deserialization the feature store performs, so a
    # field this repository renames or drops fails at load rather than at an assertion.
    reloaded = DesignSnapshotArtifact.model_validate_json(snapshot.model_dump_json())
    assert reloaded.nodes[0].content == snapshot.nodes[0].content
    assert reloaded.files[0].file_version == snapshot.files[0].file_version


def test_a_snapshot_cannot_list_one_frame_as_both_quoted_and_missing() -> None:
    """The lists are what a judge reads to know what it was not shown."""
    base = {
        "schema_version": "1.0.0",
        "workflow_id": "feature-1",
        "artifact_id": "018_design_snapshot.json",
        "producer": "design_resolver",
        "timestamp": datetime(2026, 9, 7, tzinfo=UTC),
        "metadata": {},
        "validation_status": "valid",
        "feature_id": "feature-1",
        "resolved_at": datetime(2026, 9, 7, tzinfo=UTC),
        "nodes": [
            {
                "file_key": REAL_FILE_KEY,
                "node_id": REAL_FRAME_NODE_ID,
                "content": {"name": REAL_FRAME_NAME},
            }
        ],
    }

    with pytest.raises(ValueError, match="also quotes"):
        DesignSnapshotArtifact.model_validate(
            {
                **base,
                "design_nodes_absent": [
                    {
                        "file_key": REAL_FILE_KEY,
                        "node_id": REAL_FRAME_NODE_ID,
                        "reason": "not_defined_in_file",
                    }
                ],
            }
        )


# --------------------------------------------------------------------------------------
# The detail tier is its own artifact, per repository (96- Part B)
# --------------------------------------------------------------------------------------


def detail_payload(**overrides: Any) -> dict[str, Any]:
    """One well-formed detail artifact's payload, for the validator cases below."""
    return {
        "schema_version": "1.0.0",
        "workflow_id": "feature-1",
        "artifact_id": "018_design_detail.frontend.json",
        "producer": "design_resolver",
        "timestamp": datetime(2026, 9, 10, tzinfo=UTC),
        "metadata": {},
        "validation_status": "valid",
        "feature_id": "feature-1",
        "repository_id": "frontend",
        "workstream_id": "workstream-frontend",
        "snapshot_artifact_id": "018_design_snapshot.json",
        "resolved_at": datetime(2026, 9, 10, tzinfo=UTC),
        "nodes": [
            {
                "file_key": PRODUCT_FILE_KEY,
                "node_id": PRODUCT_FORM_NODE_ID,
                "content": {"name": "Sign-in form", "layout": {"layoutMode": "VERTICAL"}},
            }
        ],
        **overrides,
    }


def test_a_detail_artifact_records_which_index_revision_it_belongs_to() -> None:
    """`snapshot_artifact_id` is the mechanism, not the provenance.

    The snapshot has revisions -- a re-resolution is a new revision and never an edit -- so
    "attempt 2 built against the same design attempt 1 did" is only answerable if each detail
    says which index it was resolved against. Required rather than defaulted, because a detail
    that omits it makes the pairing unknowable after the first refresh.
    """
    detail = DesignDetailArtifact.model_validate(detail_payload())

    assert detail.artifact_type == "design_detail"
    assert detail.snapshot_artifact_id == "018_design_snapshot.json"
    assert (detail.repository_id, detail.workstream_id) == ("frontend", "workstream-frontend")

    # And it survives the round trip a persisted artifact takes, the way the snapshot's own
    # test asserts: a field renamed or dropped fails at load rather than at an assertion.
    reloaded = DesignDetailArtifact.model_validate_json(detail.model_dump_json())
    assert reloaded.nodes[0].content == detail.nodes[0].content

    with pytest.raises(ValidationError):
        DesignDetailArtifact.model_validate(
            {key: value for key, value in detail_payload().items() if key != "snapshot_artifact_id"}
        )


def test_a_detail_cannot_list_one_frame_as_both_quoted_and_missing() -> None:
    """The snapshot's own rule, and it matters more here than there.

    This is the tier an engineer actually builds from, so a frame quoted *and* named as
    omitted is the one ambiguity that reaches code: the Engineer would read a screen it was
    shown and a sentence saying it was not shown, for the same frame. Both validators call one
    shared refusal rather than two written to agree.
    """
    for field, entry in (
        (
            "design_nodes_omitted",
            {
                "file_key": PRODUCT_FILE_KEY,
                "node_id": PRODUCT_FORM_NODE_ID,
                "reason": "over_per_node_character_bound",
                "characters": 999_999,
            },
        ),
        (
            "design_nodes_absent",
            {
                "file_key": PRODUCT_FILE_KEY,
                "node_id": PRODUCT_FORM_NODE_ID,
                "reason": "not_defined_in_file",
            },
        ),
        (
            "design_nodes_unreachable",
            {
                "file_key": PRODUCT_FILE_KEY,
                "node_id": PRODUCT_FORM_NODE_ID,
                "reason": "token_could_not_read_the_file",
            },
        ),
    ):
        with pytest.raises(ValueError, match="also quotes"):
            DesignDetailArtifact.model_validate(detail_payload(**{field: [entry]}))

    # A frame in an omission list that is *not* quoted is the ordinary honest case, and must
    # still be accepted -- that is what omit-don't-trim looks like in this tier.
    accepted = DesignDetailArtifact.model_validate(
        detail_payload(
            design_nodes_omitted=[
                {
                    "file_key": PRODUCT_FILE_KEY,
                    "node_id": PRODUCT_FRAME_NODE_ID,
                    "reason": "over_per_node_character_bound",
                    "characters": 765_949,
                }
            ]
        )
    )
    assert accepted.design_nodes_omitted[0].characters == 765_949


def test_one_detail_artifact_id_per_repository_and_a_revision_per_re_resolution() -> None:
    """Two workstreams must never land on one filename, and a refresh never overwrites.

    Keyed by repository because several of these exist per feature at once -- unlike the
    snapshot, of which there is one. Two of them sharing a filename would read as a workstream
    handed another workstream's screens, which is worse than either being missing.
    """
    first = design_detail_artifact_id("frontend", 0)
    second = design_detail_artifact_id("backend", 0)

    assert first == "018_design_detail.frontend.json"
    assert second == "018_design_detail.backend.json"
    assert first != second

    # Safety rule 4: a re-resolution is a new revision, so the detail an earlier attempt was
    # actually given stays readable.
    assert design_detail_artifact_id("frontend", 1) == "018_design_detail.frontend.revision-2.json"
    assert design_detail_artifact_id("frontend", 2) == "018_design_detail.frontend.revision-3.json"

    # The same shape the snapshot's id helper produces, which is what keeps a reader of the
    # artifact list able to tell revisions from repositories.
    assert design_snapshot_artifact_id(0) == "018_design_snapshot.json"
    assert design_snapshot_artifact_id(1) == "018_design_snapshot.revision-2.json"


def test_the_detail_artifact_survives_the_state_serialization_the_snapshot_does() -> None:
    """A new artifact absent from the `Artifact` union is dropped or refused at load.

    Asserted through the union rather than the class, because that is the deserialization a
    feature's persisted state actually performs: the state snapshot carries `artifacts` typed
    as `Artifact`, and a member missing from it either fails to validate or silently validates
    as some other artifact that happens to fit.
    """
    payload = detail_payload()
    adapter: Any = TypeAdapter(Artifact)
    restored = adapter.validate_python(payload)

    assert isinstance(restored, DesignDetailArtifact)
    assert restored.artifact_id == "018_design_detail.frontend.json"
    # Not confused with the snapshot, whose discriminant is the only thing distinguishing the
    # two once both are in the union.
    assert restored.artifact_type == "design_detail"


# --------------------------------------------------------------------------------------
# Detail is resolved for one repository's assigned frames (96- Part C)
# --------------------------------------------------------------------------------------


def product_citation(node_ids: list[str] | None = None, **overrides: Any) -> DesignReference:
    """One citation against the product file, in the URL form a browser produces."""
    query = "" if node_ids is None else "?node-id=" + ",".join(node_ids)
    return DesignReference(
        url=f"https://www.figma.com/design/{PRODUCT_FILE_KEY}/Design-Battlefield{query}",
        **overrides,
    )


def product_nodes_result(*node_ids: str, absent: tuple[str, ...] = ()) -> FigmaNodesResult:
    """A nodes answer carrying real subtrees from the committed product capture."""
    return FigmaNodesResult(
        file_key=PRODUCT_FILE_KEY,
        file_name="Design Battlefield",
        file_version="2397577240008270729",
        subtrees=tuple(product_subtree(node_id) for node_id in node_ids),
        absent_node_ids=absent,
    )


@pytest.mark.asyncio
async def test_detail_resolves_only_the_frames_one_workstream_was_assigned() -> None:
    """Two assigned frames, one artifact naming both, at build fidelity.

    The whole point of the tier: what a workstream is handed is proportional to the workstream
    and not to the feature. So the frames it was *not* assigned are not read, not quoted, and
    not paid for -- and the ones it was are quoted with the layout, typography and paint the
    index deliberately drops.
    """
    client = _ScriptedFigmaClient(
        nodes={PRODUCT_FILE_KEY: product_nodes_result(PRODUCT_FORM_NODE_ID, "74046:28752")}
    )
    detail = await resolver(client, depth_bound=MAX_DESIGN_DETAIL_DEPTH).resolve_detail(
        feature_id="feature-1",
        repository_id="frontend",
        workstream_id="workstream-frontend",
        snapshot_artifact_id="018_design_snapshot.json",
        references=[product_citation([PRODUCT_FORM_NODE_ID, "74046:28752", "74235:9125"])],
        node_ids=[PRODUCT_FORM_NODE_ID, "74046:28752"],
        artifact_id="018_design_detail.frontend.json",
    )

    assert [item.node_id for item in detail.nodes] == [PRODUCT_FORM_NODE_ID, "74046:28752"]
    assert detail.repository_id == "frontend"
    assert detail.snapshot_artifact_id == "018_design_snapshot.json"
    assert detail.design_nodes_omitted == []

    # Only the assigned frames were asked for -- the third citation was never read.
    assert client.node_calls == [(PRODUCT_FILE_KEY, [PRODUCT_FORM_NODE_ID, "74046:28752"])]

    # Build fidelity, which is what distinguishes this tier from the index. Guarded by the
    # predicate rather than by a key check, because this is the property Risk 1 turns on.
    for record in detail.nodes:
        assert design_content_is_buildable(record.content)
    assert "typography" in json.dumps(detail.nodes[0].content)
    assert detail.bounds["depth"] == MAX_DESIGN_DETAIL_DEPTH


@pytest.mark.asyncio
async def test_detail_quotes_the_frames_in_the_plans_order_not_the_providers() -> None:
    """The bound is spent in the order the workstream was told to build.

    Figma answers `nodes` as a mapping and its ordering is not a promise, so a resolution that
    trusted it would let the provider decide which of two frames a tight bound fits. The plan
    named one first; that is the one that survives.
    """
    reversed_answer = FigmaNodesResult(
        file_key=PRODUCT_FILE_KEY,
        file_name="Design Battlefield",
        file_version="1",
        # Answered smallest-last, i.e. not in the plan's order.
        subtrees=(product_subtree("74046:28752"), product_subtree(PRODUCT_FORM_NODE_ID)),
        absent_node_ids=(),
    )
    detail = await resolver(
        _ScriptedFigmaClient(nodes={PRODUCT_FILE_KEY: reversed_answer}),
        node_character_bound=MAX_DESIGN_DETAIL_NODE_CHARACTERS,
        total_character_bound=38_000,
    ).resolve_detail(
        feature_id="feature-1",
        repository_id="frontend",
        workstream_id="workstream-frontend",
        snapshot_artifact_id="018_design_snapshot.json",
        references=[product_citation([PRODUCT_FORM_NODE_ID, "74046:28752"])],
        node_ids=[PRODUCT_FORM_NODE_ID, "74046:28752"],
        artifact_id="018_design_detail.frontend.json",
    )

    # The plan's first frame is quoted; the second did not fit and is named with its size.
    assert [item.node_id for item in detail.nodes] == [PRODUCT_FORM_NODE_ID]
    assert [item.node_id for item in detail.design_nodes_omitted] == ["74046:28752"]
    assert detail.design_nodes_omitted[0].reason == "over_total_character_bound"
    assert detail.design_nodes_omitted[0].characters > 0


@pytest.mark.asyncio
async def test_detail_reports_rather_than_refuses_when_nothing_can_be_read() -> None:
    """Unlike `resolve`, this never ends the workstream over a design it can report.

    `resolve` runs before planning, where refusing costs a person a correction and no
    repository has been touched. This runs as a workstream starts, where the same refusal
    would spend a repository. So a token that cannot open the file, or no credential at all,
    produces an artifact whose lists say so -- and every prompt already carries the rule that
    what it was not shown must not be attested to.
    """
    refused = FigmaClientError(
        "figma_credential_refused_403",
        endpoint="files.nodes",
        mode=FigmaFailureMode.CREDENTIAL_REFUSED,
        error_code="figma_credential_refused_403",
        provider_status=403,
    )
    detail = await resolver(_ScriptedFigmaClient(raises=refused)).resolve_detail(
        feature_id="feature-1",
        repository_id="frontend",
        workstream_id="workstream-frontend",
        snapshot_artifact_id="018_design_snapshot.json",
        references=[product_citation([PRODUCT_FORM_NODE_ID])],
        node_ids=[PRODUCT_FORM_NODE_ID],
        artifact_id="018_design_detail.frontend.json",
    )

    assert detail.nodes == []
    assert [item.node_id for item in detail.design_nodes_unreachable] == [PRODUCT_FORM_NODE_ID]
    assert detail.design_nodes_unreachable[0].reason == "token_could_not_read_the_file"

    # A deployment with no credential at all answers the same way, for the same reason.
    async def no_client() -> FigmaDesignClient | None:
        return None

    without = await FigmaDesignResolver(client_factory=no_client).resolve_detail(
        feature_id="feature-1",
        repository_id="frontend",
        workstream_id="workstream-frontend",
        snapshot_artifact_id="018_design_snapshot.json",
        references=[product_citation([PRODUCT_FORM_NODE_ID])],
        node_ids=[PRODUCT_FORM_NODE_ID],
        artifact_id="018_design_detail.frontend.json",
    )
    assert [item.node_id for item in without.design_nodes_unreachable] == [PRODUCT_FORM_NODE_ID]


@pytest.mark.asyncio
async def test_a_transport_fault_propagates_so_the_caller_decides_what_to_do() -> None:
    """The one failure this resolver does not absorb, because the remedy differs by seam.

    A provider that did not answer may be worth waiting for; a provider that answered "no" is
    not. The retry-or-degrade decision belongs to the caller, which is the only place that
    knows whether an allowance exists to spend -- see `_resolve_design_detail`.
    """
    with pytest.raises(FigmaClientError):
        await resolver(
            _ScriptedFigmaClient(
                raises=FigmaClientError(
                    "figma_transport_429",
                    endpoint="files.nodes",
                    mode=FigmaFailureMode.TRANSPORT,
                    error_code="figma_transport_429",
                    provider_status=429,
                )
            )
        ).resolve_detail(
            feature_id="feature-1",
            repository_id="frontend",
            workstream_id="workstream-frontend",
            snapshot_artifact_id="018_design_snapshot.json",
            references=[product_citation([PRODUCT_FORM_NODE_ID])],
            node_ids=[PRODUCT_FORM_NODE_ID],
            artifact_id="018_design_detail.frontend.json",
        )


@pytest.mark.asyncio
async def test_the_deterministic_resolver_implements_detail_too() -> None:
    """Risk 4 of 96-: every test outside the live tier runs this class.

    A `resolve_detail` only `FigmaDesignResolver` had would pass the whole mock tier and fail
    nothing until a live feature ran. So the mock produces the same artifact contract -- and
    crucially content that is *buildable*, because a mock whose detail records were index-shaped
    would make the Part D guard fire on every mock feature.
    """
    detail = await DeterministicDesignResolver().resolve_detail(
        feature_id="feature-1",
        repository_id="frontend",
        workstream_id="workstream-frontend",
        snapshot_artifact_id="018_design_snapshot.json",
        references=[product_citation([PRODUCT_FORM_NODE_ID, "74046:28752"], label="Sign-in")],
        node_ids=[PRODUCT_FORM_NODE_ID],
        artifact_id="018_design_detail.frontend.json",
    )

    assert isinstance(detail, DesignDetailArtifact)
    assert [item.node_id for item in detail.nodes] == [PRODUCT_FORM_NODE_ID]
    assert detail.nodes[0].label == "Sign-in"
    assert detail.metadata["resolution_mode"] == "mock"
    assert design_content_is_buildable(detail.nodes[0].content)


def test_both_resolvers_implement_every_method_the_protocol_declares() -> None:
    """Risk 4 of 96- as a test rather than as a convention.

    Structural rather than `isinstance`, because the protocol is not `@runtime_checkable` and
    making it so to satisfy a test would be a production change for a test's benefit. Derived
    from the protocol's own members, so a *third* method added there is covered by this without
    anybody remembering to extend it -- which is the failure mode being guarded, one level up.
    """
    declared = {
        name
        for name in vars(DesignReferenceResolver)
        if not name.startswith("_") and callable(getattr(DesignReferenceResolver, name))
    }
    assert declared == {"resolve", "resolve_detail"}, "the protocol grew a method"

    for implementation in (DeterministicDesignResolver, FigmaDesignResolver):
        missing = {name for name in declared if not callable(getattr(implementation, name, None))}
        assert not missing, f"{implementation.__name__} does not implement {sorted(missing)}"


@pytest.mark.asyncio
async def test_a_whole_file_citation_still_attributes_the_frames_it_never_named() -> None:
    """A whole-file citation writes down no frame ids, and the plan assigns them anyway.

    Its frames are listed from the file at snapshot time, so they appear in no citation's
    `node_ids`. Detail resolution still has to know which file each belongs to, and this is
    the fallback that answers it -- without it, a whole-file citation would resolve to an
    empty detail artifact and the workstream would silently get no design.
    """
    detail = await resolver(
        _ScriptedFigmaClient(nodes={PRODUCT_FILE_KEY: product_nodes_result(PRODUCT_FORM_NODE_ID)})
    ).resolve_detail(
        feature_id="feature-1",
        repository_id="frontend",
        workstream_id="workstream-frontend",
        snapshot_artifact_id="018_design_snapshot.json",
        # No node ids: the person cited the whole file.
        references=[product_citation()],
        node_ids=[PRODUCT_FORM_NODE_ID],
        artifact_id="018_design_detail.frontend.json",
    )

    assert [item.node_id for item in detail.nodes] == [PRODUCT_FORM_NODE_ID]
    assert detail.nodes[0].file_key == PRODUCT_FILE_KEY


class _CountingDetailResolver(DeterministicDesignResolver):
    """The deterministic resolver, counting detail resolutions and able to fault on demand."""

    def __init__(self, *, faults: int = 0, error: BaseException | None = None) -> None:
        """Bind how many leading calls raise, and what they raise."""
        self.detail_calls: list[tuple[str, tuple[str, ...]]] = []
        self._remaining_faults = faults
        self._error = error or FigmaClientError(
            "figma_transport_429",
            endpoint="files.nodes",
            mode=FigmaFailureMode.TRANSPORT,
            error_code="figma_transport_429",
            provider_status=429,
        )

    async def resolve_detail(self, **kwargs: Any) -> DesignDetailArtifact:
        """Record the call, fault while the budget says to, then answer deterministically."""
        self.detail_calls.append((kwargs["repository_id"], tuple(kwargs["node_ids"])))
        if self._remaining_faults > 0:
            self._remaining_faults -= 1
            raise self._error
        return await super().resolve_detail(**kwargs)


async def _state_with_a_snapshot(
    *, design_nodes: list[str]
) -> tuple[Any, RepositorySpec, RepositoryWorkstreamPlan]:
    """One feature state carrying a resolved snapshot, and the workstream that reads it."""
    payload = feature_payload()
    payload["feature_id"] = "feature-detail"
    payload["prd"] = {
        **cast("dict[str, Any]", payload["prd"]),
        "design_references": [
            {
                "url": (
                    f"https://www.figma.com/design/{PRODUCT_FILE_KEY}/Design-Battlefield"
                    f"?node-id={PRODUCT_FORM_NODE_ID.replace(':', '-')}"
                )
            }
        ],
    }
    request = StartFeatureRequest.model_validate(payload)
    state = _initial_feature_state("feature-detail", request)
    snapshot = await DeterministicDesignResolver().resolve(
        feature_id="feature-detail",
        references=request.prd.design_references,
        artifact_id="018_design_snapshot.json",
    )
    state.artifacts.append(snapshot)
    repository = request.repositories[0]
    workstream = RepositoryWorkstreamPlan.model_validate(
        {
            "workstream_id": f"workstream-{repository.repository_id}",
            "repository_id": repository.repository_id,
            "role": repository.role,
            "requirement_ids": ["screen-1"],
            "scoped_requirements": [
                {
                    "requirement_id": "screen-1",
                    "acceptance_criterion_ids": ["screen-1:ac-1"],
                    "responsibility": "implements",
                }
            ],
            "out_of_scope_requirements": [],
            "shared_requirements": [],
            "responsibilities": ["Build the cited screen."],
            "task_ids": ["screen-1-task"],
            "dependency_workstream_ids": [],
            "contract_sections_consumed": [],
            "contract_sections_implemented": [],
            "acceptance_criteria": ["The screen matches the cited design."],
            "test_requirements": ["Run the configured test command."],
            "documentation_requirements": [],
            "design_nodes": design_nodes,
            "expected_files_or_areas": ["src"],
            "required": True,
        }
    )
    return state, repository, workstream


@pytest.mark.asyncio
async def test_detail_is_resolved_once_per_workstream_and_reused_by_every_attempt() -> None:
    """A design that moves under a retry is a work order that moves under a retry.

    The same rule that stopped AB-Feature-216's coder minting a `pytest` gate: what a
    workstream is judged against is derived once, before it starts, and pinned. Here the pin is
    the artifact itself, matched on this repository and on the snapshot revision -- which is
    stronger than an attempt counter, because a resume after a crash re-enters this loop with
    the counter it already had.
    """
    state, repository, workstream = await _state_with_a_snapshot(
        design_nodes=[PRODUCT_FORM_NODE_ID]
    )
    resolver_double = _CountingDetailResolver()
    orchestrator = FeatureWorkflowOrchestrator(design_resolver=resolver_double)

    first = await orchestrator._resolve_design_detail(
        state, repository=repository, workstream=workstream
    )
    assert first is not None
    assert [item.node_id for item in first.nodes] == [PRODUCT_FORM_NODE_ID]
    assert len(resolver_double.detail_calls) == 1

    # Re-entered for the next attempt: the artifact is found and nothing is read again.
    again = await orchestrator._resolve_design_detail(
        state, repository=repository, workstream=workstream
    )
    assert again is first
    assert len(resolver_double.detail_calls) == 1, "attempt 2 re-resolved the design"

    # And exactly one artifact was appended, not one per attempt.
    details = [item for item in state.artifacts if isinstance(item, DesignDetailArtifact)]
    assert len(details) == 1
    assert details[0].artifact_id == f"018_design_detail.{repository.repository_id}.json"


@pytest.mark.asyncio
async def test_a_workstream_the_design_does_not_apply_to_resolves_nothing_at_all() -> None:
    """No assignment, no artifact, no call -- byte-identical to a feature that cited nothing.

    The gate every part of this item is held to. A repository the design says nothing about
    must not gain a prompt block, an artifact or an outbound call because some *other*
    workstream of the same feature had frames.
    """
    state, repository, workstream = await _state_with_a_snapshot(design_nodes=[])
    resolver_double = _CountingDetailResolver()
    orchestrator = FeatureWorkflowOrchestrator(design_resolver=resolver_double)

    assert (
        await orchestrator._resolve_design_detail(
            state, repository=repository, workstream=workstream
        )
        is None
    )
    assert resolver_double.detail_calls == []
    assert not [item for item in state.artifacts if isinstance(item, DesignDetailArtifact)]


@pytest.mark.asyncio
async def test_a_transient_fault_is_retried_and_then_degrades_instead_of_killing_the_loop() -> None:
    """An exception here would end the repository having spent none of its retries.

    That is the -072 shape: this runs in `_run_one_child` *before* its loop, so anything raised
    escapes to the fan-out, which records the workstream as failed. So a transient fault buys a
    small local allowance, and a provider that is still not answering produces an artifact
    saying the frames were unreachable -- which every prompt already knows how to read.

    AB-Feature-227 is the case in hand: a Figma rate limit whose window is hours cannot be
    waited out by any allowance sized for a blip, and holding a workstream hostage to it is
    worse than telling it what it did not get.
    """
    state, repository, workstream = await _state_with_a_snapshot(
        design_nodes=[PRODUCT_FORM_NODE_ID]
    )
    # One fault, then an answer: the allowance covers the blip and the design is resolved.
    recovers = _CountingDetailResolver(faults=1)
    orchestrator = FeatureWorkflowOrchestrator(design_resolver=recovers)
    resolved = await orchestrator._resolve_design_detail(
        state, repository=repository, workstream=workstream
    )
    assert resolved is not None
    assert [item.node_id for item in resolved.nodes] == [PRODUCT_FORM_NODE_ID]
    assert len(recovers.detail_calls) == 2, "the fault was retried once"

    # A provider that never answers: the allowance runs out and the frames are reported.
    state, repository, workstream = await _state_with_a_snapshot(
        design_nodes=[PRODUCT_FORM_NODE_ID]
    )
    never = _CountingDetailResolver(faults=99)
    degraded = await FeatureWorkflowOrchestrator(design_resolver=never)._resolve_design_detail(
        state, repository=repository, workstream=workstream
    )

    assert degraded is not None, "the workstream must not be ended by this"
    assert degraded.nodes == []
    assert [item.node_id for item in degraded.design_nodes_unreachable] == [PRODUCT_FORM_NODE_ID]
    assert degraded.metadata["resolution_mode"] == "unreachable"
    # Bounded: it did not retry forever against a provider that is down.
    assert len(never.detail_calls) == 3, "two retries beyond the first call, then degrade"


@pytest.mark.asyncio
async def test_a_deterministic_provider_answer_is_not_retried_at_this_seam_either() -> None:
    """A refusal answers the same way every time, so an allowance buys nothing.

    The rule AB-Feature-172 established by spending nine calls proving it on one 400, applied
    here: only a *transient* fault is worth the retry. A credential the provider refused
    degrades on the first answer rather than after three.
    """
    state, repository, workstream = await _state_with_a_snapshot(
        design_nodes=[PRODUCT_FORM_NODE_ID]
    )
    refused = _CountingDetailResolver(
        faults=99,
        error=FigmaClientError(
            "figma_credential_refused_403",
            endpoint="files.nodes",
            mode=FigmaFailureMode.CREDENTIAL_REFUSED,
            error_code="figma_credential_refused_403",
            provider_status=403,
        ),
    )
    detail = await FeatureWorkflowOrchestrator(design_resolver=refused)._resolve_design_detail(
        state, repository=repository, workstream=workstream
    )

    assert detail is not None
    assert [item.node_id for item in detail.design_nodes_unreachable] == [PRODUCT_FORM_NODE_ID]
    assert len(refused.detail_calls) == 1, "a refusal was retried"


def test_an_unreadable_design_source_still_names_every_assigned_frame() -> None:
    """The degradation the child loop falls back on, built from the snapshot's own knowledge.

    Provenance comes from the snapshot rather than the citations because the snapshot is the
    authority on what a frame id is -- the planner may only assign frames it mentions. A frame
    the snapshot knows only as *omitted* is still named here with its file and its label, which
    is what lets the prompt say which screen it did not get.
    """
    snapshot = DesignSnapshotArtifact.model_validate(
        {
            "schema_version": "1.0.0",
            "workflow_id": "feature-1",
            "artifact_id": "018_design_snapshot.json",
            "producer": "design_resolver",
            "timestamp": datetime(2026, 9, 10, tzinfo=UTC),
            "metadata": {},
            "validation_status": "valid",
            "feature_id": "feature-1",
            "resolved_at": datetime(2026, 9, 10, tzinfo=UTC),
            "nodes": [
                {
                    "file_key": PRODUCT_FILE_KEY,
                    "node_id": PRODUCT_FORM_NODE_ID,
                    "label": "Sign-in form",
                    "content": {"name": "Sign-in form"},
                }
            ],
            "design_nodes_omitted": [
                {
                    "file_key": PRODUCT_FILE_KEY,
                    "node_id": PRODUCT_FRAME_NODE_ID,
                    "label": "Creative studio",
                    "reason": "over_per_node_character_bound",
                    "characters": 765_949,
                }
            ],
        }
    )

    detail = unreachable_design_detail(
        feature_id="feature-1",
        repository_id="frontend",
        workstream_id="workstream-frontend",
        snapshot_artifact_id=snapshot.artifact_id,
        snapshot=snapshot,
        node_ids=[PRODUCT_FORM_NODE_ID, PRODUCT_FRAME_NODE_ID],
        artifact_id="018_design_detail.frontend.json",
    )

    assert detail.nodes == []
    named = {item.node_id: item for item in detail.design_nodes_unreachable}
    assert set(named) == {PRODUCT_FORM_NODE_ID, PRODUCT_FRAME_NODE_ID}
    # Including the one the snapshot itself could only report as omitted.
    assert named[PRODUCT_FRAME_NODE_ID].label == "Creative studio"
    assert named[PRODUCT_FRAME_NODE_ID].file_key == PRODUCT_FILE_KEY
    assert detail.metadata["resolution_mode"] == "unreachable"


# --------------------------------------------------------------------------------------
# A whole-file citation resolves to the file's top-level frames, bounded and reported
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_whole_file_citation_resolves_to_the_top_level_frames() -> None:
    """Legitimate, and answered through the same nodes endpoint at the document root."""
    frames = FigmaTopLevelResult(
        file_key=REAL_FILE_KEY,
        file_name="Untitled",
        file_version="2315052197036992991",
        frames=(
            FigmaTopLevelFrame(
                node_id=REAL_FRAME_NODE_ID,
                name=REAL_FRAME_NAME,
                node_type="FRAME",
                canvas_name="Page 1",
            ),
        ),
        frames_beyond_bound=3,
    )
    client = _ScriptedFigmaClient(
        nodes={REAL_FILE_KEY: one_file([captured_subtree()])}, top_level=frames
    )

    snapshot = await resolver(client).resolve(
        feature_id="feature-1",
        references=[citation(None)],
        artifact_id=FEATURE_ARTIFACT_FILENAMES["design_snapshot"],
    )

    assert client.top_level_calls == [REAL_FILE_KEY]
    assert [item.node_id for item in snapshot.nodes] == [REAL_FRAME_NODE_ID]
    # The frame's own name is used when the author gave no label, which is more use to a
    # reader than a node id.
    assert snapshot.nodes[0].label == REAL_FRAME_NAME


# --------------------------------------------------------------------------------------
# Mock mode, and the step's place in the sequence
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_deterministic_resolver_produces_the_same_contract_with_no_network() -> None:
    """Mock mode exercises the step's place in the order without Figma existing."""
    snapshot = await DeterministicDesignResolver().resolve(
        feature_id="feature-mock",
        references=[citation([REAL_FRAME_NODE_ID], label="mock frame")],
        artifact_id=FEATURE_ARTIFACT_FILENAMES["design_snapshot"],
    )

    assert isinstance(snapshot, DesignSnapshotArtifact)
    assert snapshot.metadata["resolution_mode"] == "mock"
    assert [item.node_id for item in snapshot.nodes] == [REAL_FRAME_NODE_ID]
    assert snapshot.nodes[0].label == "mock frame"
    assert snapshot.files[0].file_version == "mock-version"
    assert snapshot.characters_selected > 0
    # A whole-file citation resolves too: a mock has no file to list, so it stands one in.
    whole = await DeterministicDesignResolver().resolve(
        feature_id="feature-mock",
        references=[citation(None)],
        artifact_id=FEATURE_ARTIFACT_FILENAMES["design_snapshot"],
    )
    assert len(whole.nodes) == 1


def test_the_fetch_operation_type_belongs_to_the_pre_coding_block() -> None:
    """It records; it is never a recovery input, which is that block's whole property."""
    assert ExternalOperationType.FETCH_DESIGN_REFERENCE.value == "fetch_design_reference"
    from services.execution_records import PLANNING_CALL_OPERATION_TYPES, ExecutionStage

    assert ExternalOperationType.FETCH_DESIGN_REFERENCE in PLANNING_CALL_OPERATION_TYPES
    assert ExecutionStage.DESIGN_SNAPSHOT.value == "design_snapshot"


def test_the_snapshot_filename_follows_the_last_one() -> None:
    """`017_design_conflict.json` was the last; a new artifact takes the next number."""
    assert FEATURE_ARTIFACT_FILENAMES["design_snapshot"] == "018_design_snapshot.json"
    assert FEATURE_ARTIFACT_FILENAMES["design_conflict"] == "017_design_conflict.json"


def _accepted(references: list[dict[str, Any]] | None = None) -> Any:
    """The state a feature has the moment it is accepted, from the real acceptance path.

    Built through `StartFeatureRequest` and `_initial_feature_state` rather than by hand, for
    `test_feature_steps.py`'s reason: a hand-written snapshot agrees with whatever the person
    writing it believed, and this one is what the control plane actually persists -- including
    the normalization the citation went through on the way in.
    """
    payload = feature_payload()
    if references is not None:
        payload["prd"] = {**payload["prd"], "design_references": references}
    return _initial_feature_state(
        "feature-design-step", StartFeatureRequest.model_validate(payload)
    )


def test_a_citation_free_feature_is_never_offered_the_resolution_step() -> None:
    """Safety rule 3, at the one place that decides what a feature does next.

    `next_step` is total, so this is the whole claim: a feature that cited nothing is never
    named this step, and its first step is the one it has always been.
    """
    decision = next_step(_accepted())

    assert decision.step is FeatureStep.ANALYZE_PRD
    assert "product manager runs first" in decision.explanation


def test_a_feature_that_cites_a_design_resolves_it_before_the_product_manager() -> None:
    """Before, not after: a design that arrives after the requirements can only contradict them.

    And the artifact's absence is the evidence the step is owed -- there is deliberately no
    stored current step -- so once a snapshot exists the feature moves on and never re-reads.
    """
    state = _accepted(
        [{"url": f"https://www.figma.com/design/{REAL_FILE_KEY}/Untitled?node-id=10-11"}]
    )

    decision = next_step(state)

    assert decision.step is FeatureStep.RESOLVE_DESIGN
    assert "cites a design" in decision.explanation

    # The citation was normalized on the way in, which is what the resolver reads.
    prd = next(item for item in state.artifacts if item.artifact_type == "prd")
    assert prd.design_references[0].file_key == REAL_FILE_KEY
    assert prd.design_references[0].node_ids == ["10:11"]


@pytest.mark.asyncio
async def test_the_snapshot_is_what_stops_the_step_being_owed_again() -> None:
    """It resumes from its artifact like every other pre-coding stage."""
    state = _accepted(
        [{"url": f"https://www.figma.com/design/{REAL_FILE_KEY}/Untitled?node-id=10-11"}]
    )
    prd = next(item for item in state.artifacts if item.artifact_type == "prd")

    snapshot = await DeterministicDesignResolver().resolve(
        feature_id=state.feature_id,
        references=prd.design_references,
        artifact_id=design_snapshot_artifact_id(0),
    )
    state.artifacts.append(snapshot)

    assert next_step(state).step is FeatureStep.ANALYZE_PRD


@pytest.mark.asyncio
async def test_a_mock_feature_that_cites_a_design_resolves_it_and_journals_nothing() -> None:
    """The step's place in the sequence, exercised end to end with no adapter at all.

    Mock mode runs a deterministic orchestrator with no adapters and no LLM client, and this
    is what makes "before the product manager" a fact about the run order rather than about
    `next_step` in isolation.
    """
    state = _accepted(
        [{"url": f"https://www.figma.com/design/{REAL_FILE_KEY}/Untitled?node-id=10-11"}]
    )
    orchestrator = FeatureWorkflowOrchestrator()

    final = await orchestrator.start(
        state, credentials=RequestScopedCredentials(openai_api_key=None, github_token=None)
    )

    kinds = [item.artifact_type for item in final.artifacts]
    assert "design_snapshot" in kinds
    # Resolved before the technical PRD was written, which is the whole point of the ordering.
    assert kinds.index("design_snapshot") < kinds.index("technical_prd")
    snapshot = next(item for item in final.artifacts if item.artifact_type == "design_snapshot")
    assert snapshot.artifact_id == "018_design_snapshot.json"
    assert snapshot.metadata["resolution_mode"] == "mock"


@pytest.mark.asyncio
async def test_a_citation_free_mock_feature_produces_no_snapshot_at_all() -> None:
    """No extra artifact, no extra step, and nothing new in its artifact count."""
    orchestrator = FeatureWorkflowOrchestrator()

    final = await orchestrator.start(
        _accepted(), credentials=RequestScopedCredentials(openai_api_key=None, github_token=None)
    )

    assert [item for item in final.artifacts if item.artifact_type == "design_snapshot"] == []


def test_no_bound_bites_on_the_real_file_that_calibrated_them() -> None:
    """A bound that bites on ordinary input is a bound that routinely hides the design.

    So the numbers in `artifacts/design_references.py` are asserted against the committed live
    capture rather than merely written beside it. Measured here, through the shipped
    extraction: the largest real frame is 8,994 characters -- 22% of the per-node bound -- the
    whole file 16,941 (21% of the total), the deepest natural depth 4 against a bound of 8, and
    584 characters per node. If somebody "tunes" a bound down, this is what stops it.
    """
    payload = captured_file()
    measured = []
    for canvas in payload["document"]["children"]:
        for frame in canvas.get("children") or []:
            subtree = FigmaNodeSubtree(
                node_id=frame["id"],
                document=frame,
                styles=payload.get("styles") or {},
                components=payload.get("components") or {},
                component_sets=payload.get("componentSets") or {},
            )
            content, nodes, depth = extract_design_node(subtree)
            measured.append(
                (
                    len(json.dumps(content, indent=2, sort_keys=True, ensure_ascii=False)),
                    nodes,
                    depth,
                )
            )

    assert len(measured) == 3, "the captured file has three top-level frames"
    largest = max(item[0] for item in measured)
    total = sum(item[0] for item in measured)
    deepest = max(item[2] for item in measured)

    assert largest == 9_578, "the measurement the comment records, so a drift is visible here"
    assert total == 18_002
    assert deepest == 4
    assert largest < MAX_DESIGN_DETAIL_NODE_CHARACTERS
    assert total < MAX_DESIGN_DETAIL_TOTAL_CHARACTERS
    assert deepest <= MAX_DESIGN_DEPTH
    assert len(measured) <= MAX_DESIGN_NODES
    # Headroom rather than a trim: a bound within a factor of two of real input is one that
    # will bite on the next slightly denser screen.
    assert largest * 4 < MAX_DESIGN_DETAIL_NODE_CHARACTERS
    assert total * 4 < MAX_DESIGN_DETAIL_TOTAL_CHARACTERS


def test_no_bound_bites_on_the_product_screen_that_recalibrated_them() -> None:
    """The same property, on the file that proved the first calibration wrong (96- Part E).

    The bounds above were set against a demo file whose largest frame is 8,994 characters. This
    file's ordinary screens are 119,049 to 196,447 at depth 8, so against the *old* numbers --
    40,000 per node, 80,000 in total -- every one of them was omitted whole and every prompt was
    correctly told the design had not been shown. That is the bug 96- exists to fix, and this is
    the test that stops it coming back: if somebody tunes the detail bounds back down toward the
    demo file's sizes, a real screen stops resolving and this fails.

    Measured at `_MIN_DETAIL_DEPTH` rather than at `MAX_DESIGN_DETAIL_DEPTH`, because since the
    fallback landed the latter is a *ceiling* and not the depth every frame renders at: this
    frame is 754,605 characters at 14 and falls back to 10, so 10 is the rendering the bounds
    have to accommodate and 10 is what this pins.
    """
    measured = {
        node_id: len(
            json.dumps(
                extract_design_node(product_subtree(node_id), depth_bound=_MIN_DETAIL_DEPTH)[0],
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
        )
        for node_id in (PRODUCT_FRAME_NODE_ID, PRODUCT_FORM_NODE_ID)
    }

    # The numbers the comment in `design_references.py` records.
    assert measured[PRODUCT_FRAME_NODE_ID] == 286_302
    assert measured[PRODUCT_FORM_NODE_ID] == 37_139

    # A whole product screen now fits at build fidelity, which it did not before this item.
    assert measured[PRODUCT_FRAME_NODE_ID] < MAX_DESIGN_DETAIL_NODE_CHARACTERS
    assert measured[PRODUCT_FRAME_NODE_ID] < MAX_DESIGN_DETAIL_TOTAL_CHARACTERS
    # And two of them fit together, which is what one workstream realistically builds.
    assert sum(measured.values()) < MAX_DESIGN_DETAIL_TOTAL_CHARACTERS

    # The depth bound is the one that must not silently bite: the real tree is 17 deep, so a
    # frame resolved for building reports the cut rather than reading as a frame that ends.
    content, nodes, depth = extract_design_node(product_subtree(), depth_bound=_MIN_DETAIL_DEPTH)
    assert (nodes, depth) == (166, 10)
    assert MAX_DESIGN_DETAIL_DEPTH > MAX_DESIGN_DEPTH, "detail must see deeper than the index"
    assert "children_beyond_depth_bound" in json.dumps(content)


@pytest.mark.asyncio
async def test_a_frame_over_even_the_raised_bound_is_still_omitted_whole_and_sized() -> None:
    """Omit-don't-trim survives the raise, which is the point of raising rather than trimming.

    A bound that is larger is still a bound, and what it does when it bites is unchanged: the
    frame is omitted **whole**, named, and its measured size recorded, so every judge reads
    "you were not shown this, and here is how big it was" rather than inferring a screen with
    no children from a layout that stops.
    """
    oversized = product_subtree()
    # Measured the way the resolver renders at its fallback floor: build fidelity to
    # `_MIN_DETAIL_DEPTH` with an index tail to the leaves. That is the smallest rendering the
    # resolver will produce, so it is the size an omission at this bound has to name.
    content, _, _ = extract_design_node(
        oversized, depth_bound=MAX_DESIGN_DETAIL_INDEX_DEPTH, build_depth=_MIN_DETAIL_DEPTH
    )
    characters = len(json.dumps(content, indent=2, sort_keys=True, ensure_ascii=False))
    assert characters == 362_771, "the frame this test needs to be over a lowered bound"

    # Asserted on the **detail** resolution, which is the tier these bounds govern and the tier
    # an engineer builds from -- so it is the one where trimming instead of omitting would put
    # a half-drawn screen in front of the role that writes the CSS.
    client = _ScriptedFigmaClient(
        nodes={
            PRODUCT_FILE_KEY: FigmaNodesResult(
                file_key=PRODUCT_FILE_KEY,
                file_name="Design Battlefield",
                file_version="1",
                subtrees=(oversized, product_subtree(PRODUCT_FORM_NODE_ID)),
                absent_node_ids=(),
            )
        }
    )
    detail = await resolver(
        client,
        node_character_bound=100_000,
        depth_bound=MAX_DESIGN_DETAIL_DEPTH,
    ).resolve_detail(
        feature_id="feature-1",
        repository_id="frontend",
        workstream_id="workstream-frontend",
        snapshot_artifact_id="018_design_snapshot.json",
        references=[product_citation([PRODUCT_FRAME_NODE_ID, PRODUCT_FORM_NODE_ID])],
        node_ids=[PRODUCT_FRAME_NODE_ID, PRODUCT_FORM_NODE_ID],
        artifact_id="018_design_detail.frontend.json",
    )

    omitted = {item.node_id: item for item in detail.design_nodes_omitted}
    assert set(omitted) == {PRODUCT_FRAME_NODE_ID}
    assert omitted[PRODUCT_FRAME_NODE_ID].reason == "over_per_node_character_bound"
    assert omitted[PRODUCT_FRAME_NODE_ID].characters == characters, "the size is named"

    # And it cost only itself: the smaller frame beside it is still quoted whole. Nothing is
    # trimmed, and nothing later in the plan's order is dropped for being later.
    assert [item.node_id for item in detail.nodes] == [PRODUCT_FORM_NODE_ID]
    assert design_content_is_buildable(detail.nodes[0].content)


# --------------------------------------------------------------------------------------
# The console's preview: rendered on demand, pinned, and never stored
# --------------------------------------------------------------------------------------


class _PreviewClient(FigmaDesignClient):
    """Record what a preview was asked for and answer with bytes, or refuse."""

    def __init__(self, *, raises: FigmaClientError | None = None) -> None:
        """Bind whether this client renders or refuses."""
        self.renders: list[tuple[str, str, str]] = []
        self.fetched: list[str] = []
        self._raises = raises

    async def render_preview(self, file_key: str, node_id: str, *, version: str) -> str:
        """Record the render, including the version it was pinned to."""
        self.renders.append((file_key, node_id, version))
        if self._raises is not None:
            raise self._raises
        return "https://figma-alpha-api.example/images/abc"

    async def fetch_rendered_bytes(self, url: str) -> bytes:
        """Return a minimal real PNG header, which is all the endpoint passes through."""
        self.fetched.append(url)
        return b"\x89PNG\r\n\x1a\n" + b"pretend-pixels"


async def _feature_with_a_snapshot(app: Any, headers: dict[str, str]) -> str:
    """Start a mock feature citing the real frame, and return its id.

    Mock mode, so the resolution runs through the deterministic resolver and the endpoint is
    exercised against a snapshot the platform actually wrote rather than one placed in state.
    """
    from tests.support import settle

    payload = feature_payload()
    payload["feature_id"] = "feature-preview"
    payload["prd"] = {
        **cast("dict[str, Any]", payload["prd"]),
        "design_references": [
            {"url": f"https://www.figma.com/design/{REAL_FILE_KEY}/Untitled?node-id=10-11"}
        ],
    }
    async with _client(app) as http:
        started = await http.post("/features/start", headers=headers, json=payload)
        assert started.status_code == 201, started.text
    await settle(app)
    return str(started.json()["feature_id"])


@pytest.mark.asyncio
async def test_a_preview_is_rendered_on_demand_and_pinned_to_the_recorded_version() -> None:
    """No URL is stored anywhere, and the render is pinned to what the snapshot recorded.

    Figma's images endpoint renders the file's *current* state unless told otherwise, so an
    unpinned re-render would show a design that had changed under an unchanged snapshot -- the
    console quietly disagreeing with the text every judge was given.
    """
    app, directory, _ = await _app_with_design_source()
    headers = await _operator_headers(directory)
    preview = _PreviewClient()

    async def factory() -> FigmaDesignClient | None:
        return preview

    app.state.figma_client_factory = factory
    async with _client(app) as http:
        await http.put("/design-source", headers=headers, json={"enabled": True})
    feature_id = await _feature_with_a_snapshot(app, headers)

    async with _client(app) as http:
        response = await http.get(
            f"/features/{feature_id}/design-preview",
            headers=headers,
            params={"node_id": "10:11"},
        )
        artifacts = await http.get(
            f"/features/{feature_id}/artifacts?artifact_type=design_snapshot", headers=headers
        )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content.startswith(b"\x89PNG")
    # Private, and short: a design is this deployment's, and a slow preview is a preview.
    assert response.headers["cache-control"] == "private, max-age=300"
    stored = artifacts.json()["artifacts"][0]["payload"]
    assert preview.renders == [(REAL_FILE_KEY, "10:11", stored["files"][0]["file_version"])]
    # The render URL exists for the duration of one request and reaches nothing durable.
    assert preview.fetched == ["https://figma-alpha-api.example/images/abc"]
    assert "figma-alpha-api" not in json.dumps(stored)


@pytest.mark.asyncio
async def test_a_frame_the_snapshot_does_not_quote_has_no_preview() -> None:
    """Said, with which of the three answers it was. A 404 with a sentence, not a blank image."""
    app, directory, _ = await _app_with_design_source()
    headers = await _operator_headers(directory)

    async def factory() -> FigmaDesignClient | None:
        return _PreviewClient()

    app.state.figma_client_factory = factory
    async with _client(app) as http:
        await http.put("/design-source", headers=headers, json={"enabled": True})
    feature_id = await _feature_with_a_snapshot(app, headers)

    async with _client(app) as http:
        response = await http.get(
            f"/features/{feature_id}/design-preview",
            headers=headers,
            params={"node_id": "999:999"},
        )

    assert response.status_code == 404
    assert "does not contain the frame 999:999" in response.json()["detail"]
    assert "says which of those happened" in response.json()["detail"]


@pytest.mark.asyncio
async def test_an_expired_render_costs_a_picture_and_never_the_design() -> None:
    """A render URL expires. The snapshot's text is what every judge was given, and it stands."""
    app, directory, _ = await _app_with_design_source()
    headers = await _operator_headers(directory)
    expired = FigmaClientError(
        "expired",
        mode=FigmaFailureMode.TRANSPORT,
        error_code="figma_render_unavailable_403",
        endpoint="images.render",
        provider_status=403,
    )

    async def factory() -> FigmaDesignClient | None:
        return _PreviewClient(raises=expired)

    app.state.figma_client_factory = factory
    async with _client(app) as http:
        await http.put("/design-source", headers=headers, json={"enabled": True})
    feature_id = await _feature_with_a_snapshot(app, headers)

    async with _client(app) as http:
        response = await http.get(
            f"/features/{feature_id}/design-preview",
            headers=headers,
            params={"node_id": "10:11"},
        )
        # And the text is still there.
        artifacts = await http.get(
            f"/features/{feature_id}/artifacts?artifact_type=design_snapshot", headers=headers
        )

    assert response.status_code == 504
    detail = response.json()["detail"]
    assert "figma_render_unavailable_403" in detail
    assert "snapshot's text is unaffected" in detail
    assert artifacts.json()["artifacts"][0]["payload"]["nodes"]


@pytest.mark.asyncio
async def test_a_deployment_that_cannot_render_says_so_rather_than_failing() -> None:
    """No Figma credential is a state. The design's text does not depend on a picture."""
    app, directory, _ = await _app_with_design_source()
    headers = await _operator_headers(directory)
    async with _client(app) as http:
        await http.put("/design-source", headers=headers, json={"enabled": True})
    feature_id = await _feature_with_a_snapshot(app, headers)
    app.state.figma_client_factory = None

    async with _client(app) as http:
        response = await http.get(
            f"/features/{feature_id}/design-preview",
            headers=headers,
            params={"node_id": "10:11"},
        )

    assert response.status_code == 503
    assert "cannot render design previews" in response.json()["detail"]


@pytest.mark.asyncio
async def test_a_render_url_off_figmas_hosts_is_refused_before_anything_is_fetched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The render URL came out of a provider response body, so it is validated, not trusted.

    `fetch_rendered_bytes` GETs whatever it is handed, which is a server-side request forgery
    the moment that body says `http://169.254.169.254/`. Refused without a request: the stub
    below stands in for `httpx` and fails the test if anything touches it.
    """
    import sys

    from adapters.figma_adapter import HttpxFigmaDesignClient

    class _NoNetwork:
        def __getattr__(self, name: str) -> Any:
            raise AssertionError(f"a refused render URL must not be fetched (asked for {name})")

    monkeypatch.setitem(sys.modules, "httpx", cast("Any", _NoNetwork()))
    client = HttpxFigmaDesignClient("figd_test_token")

    refused_urls = [
        # Plain HTTP, even on a legitimate host: the bytes would travel unprotected.
        "http://figma-alpha-api.s3.us-west-2.amazonaws.com/images/abc.png",
        # The cloud metadata service, which is what an SSRF is usually for.
        "http://169.254.169.254/latest/meta-data/",
        "https://169.254.169.254/latest/meta-data/",
        # An internal host.
        "https://internal-billing.local/render.png",
        # A lookalike that merely ends in the right letters without the dot.
        "https://evilamazonaws.com/render.png",
        "not a url at all",
    ]
    for url in refused_urls:
        with pytest.raises(FigmaClientError) as refused:
            await client.fetch_rendered_bytes(url)
        assert refused.value.error_code == "figma_render_url_refused", url
        # Deterministic: the same URL is refused the same way every time.
        assert refused.value.retryable is False, url
        # And the URL itself is not quoted into the message, which becomes durable state.
        assert url not in str(refused.value), url


@pytest.mark.asyncio
async def test_a_render_answer_that_is_not_a_png_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 200 that is not a PNG is an error page or a bucket answer, never `image/png`."""
    import sys
    import types

    class _Response:
        def __init__(self, content: bytes) -> None:
            self.status_code = 200
            self.content = content

    bodies: list[bytes] = [b"<html>AccessDenied</html>"]

    class _AsyncClient:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> _AsyncClient:
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def get(self, url: str, **_kwargs: Any) -> _Response:
            del url
            return _Response(bodies[0])

    monkeypatch.setitem(
        sys.modules, "httpx", cast("Any", types.SimpleNamespace(AsyncClient=_AsyncClient))
    )
    from adapters.figma_adapter import HttpxFigmaDesignClient

    client = HttpxFigmaDesignClient("figd_test_token")
    render_url = "https://figma-alpha-api.s3.us-west-2.amazonaws.com/images/abc.png"

    with pytest.raises(FigmaClientError) as refused:
        await client.fetch_rendered_bytes(render_url)
    assert refused.value.error_code == "figma_render_not_a_png"
    assert refused.value.retryable is False

    # And a genuine PNG on an allowed host comes through byte for byte.
    bodies[0] = b"\x89PNG\r\n\x1a\n" + b"pixels"
    assert await client.fetch_rendered_bytes(render_url) == b"\x89PNG\r\n\x1a\npixels"


def test_only_the_resolver_and_the_console_may_render_a_frame() -> None:
    """What Safety rule 1 became once the adapter could carry an image.

    The rule was written as: "the model cannot be shown an image, and this item must not
    pretend otherwise ... the design reaches an agent as text or not at all", and it said of
    itself that if *89 -- A Picture Is Part Of The Requirement* landed, "this rule survives as a
    decision rather than a constraint". That item landed: `LLMClient.respond` takes `images`
    and the product manager already sends them. So this is now a decision, and the decision has
    been changed -- a design's own artwork reaches the Engineer, because an image reaching it as
    `{"type": "IMAGE"}` beside a width produced a screen of empty boxes on AB-Feature-231.

    What survives unchanged is the property that actually protected anything: **exactly two
    places may render a frame.** The console route, for a browser, and the resolver, once per
    workstream, pinned to the file version the snapshot recorded. Not `agents/`, not
    `workflows/`, not `tools/` -- an agent that could render on demand would re-render a design
    mid-feature and defeat Safety rule 4's resolve-once guarantee, which is the rule that keeps
    attempt 2 building against the same design attempt 1 did.
    """
    root = Path(__file__).resolve().parent.parent
    allowed = {"services/design_resolution.py"}
    callers = sorted(
        str(path.relative_to(root))
        for directory in ("agents", "workflows", "services", "tools")
        for path in (root / directory).rglob("*.py")
        if "render_preview(" in path.read_text() or "fetch_rendered_bytes(" in path.read_text()
    )

    assert set(callers) <= allowed, (
        "a frame may be rendered only by the console route and the resolver; "
        f"also called from {sorted(set(callers) - allowed)}"
    )


def _parent_snapshot(feature_id: str) -> DesignSnapshotArtifact:
    """One resolved snapshot at parent scope, carrying the feature id the PM's state uses."""
    return DesignSnapshotArtifact.model_validate(
        {
            "schema_version": "1.0.0",
            "workflow_id": feature_id,
            "artifact_id": "018_design_snapshot.json",
            "producer": "design_resolver",
            "timestamp": datetime(2026, 9, 12, tzinfo=UTC),
            "metadata": {},
            "validation_status": "valid",
            "feature_id": feature_id,
            "resolved_at": datetime(2026, 9, 12, tzinfo=UTC),
            "nodes": [
                {
                    "file_key": PRODUCT_FILE_KEY,
                    "node_id": PRODUCT_FORM_NODE_ID,
                    "label": "Login page",
                    "content": {"name": "Login page", "text": "Sign in to continue"},
                }
            ],
        }
    )


@pytest.mark.asyncio
async def test_the_product_manager_is_shown_the_design_it_derives_requirements_from(
    tmp_path: Path,
) -> None:
    """The role whose job is reading the frames was the only role never given them.

    `LiveFeatureProductManager` built the agent's state as `[prd]`, and the agent finds a
    design through `design_snapshot_in_state`, which reads exactly that list -- so the block
    rendered empty and the model was told nothing. Verified live on AB-Feature-228: a
    1,831-token prompt against 77,753 characters of resolved index, producing a technical PRD
    that quoted the node id from the citation and no element from inside the frame.

    Asserted on the state the agent is handed, not on the call: passing the snapshot into a
    method that then drops it would satisfy a call-level assertion and change nothing.
    """
    from services.feature_runtime import LiveFeatureProductManager

    seen: list[list[Any]] = []

    class _CapturingAgent:
        async def run(self, state: Any) -> dict[str, Any]:
            seen.append(list(state["artifacts"]))
            return {
                "artifacts": [
                    create_artifact(
                        TechnicalPRDArtifact,
                        workflow_id=state["workflow_id"],
                        artifact_id=ARTIFACT_FILENAMES["technical_prd"],
                        producer="product_manager",
                        metadata={},
                        payload={
                            "title": "t",
                            "solution_summary": "s",
                            "functional_requirements": [
                                {
                                    "requirement_id": "FR-001",
                                    "description": "Build the login page to the cited design.",
                                    "priority": "must",
                                    "acceptance_criteria": ["It matches the frame."],
                                    "dependencies": [],
                                }
                            ],
                            "non_functional_requirements": [],
                            "data_requirements": [],
                            "integration_requirements": [],
                            "security_requirements": [],
                            "assumptions": [],
                            "unresolved_questions": [],
                        },
                    )
                ]
            }

    feature_id = "feature-design-reaches-the-pm"
    prd = create_artifact(
        PRDArtifact,
        workflow_id=feature_id,
        artifact_id=ARTIFACT_FILENAMES["prd"],
        producer="intake",
        metadata={},
        payload={"title": "Login page", "problem_statement": "Build it to the design."},
    )
    manager = LiveFeatureProductManager(cast(Any, _CapturingAgent()), tmp_path)
    snapshot = _parent_snapshot(feature_id)

    await manager.create_technical_prd(feature_id=feature_id, prd=prd, design_snapshot=snapshot)

    handed = seen[-1]
    assert any(isinstance(item, DesignSnapshotArtifact) for item in handed), (
        "the product manager's state carried no design snapshot, so its prompt renders none"
    )
    # The agent's own lookup is the thing that has to succeed, and it filters on workflow id --
    # so an artifact present under the wrong one would pass the check above and still render
    # nothing. This is the assertion that would have caught AB-Feature-228.
    from agents.shared.design_snapshot import design_request_context, design_snapshot_in_state

    found = design_snapshot_in_state(cast(Any, {"artifacts": handed, "workflow_id": feature_id}))
    assert found is not None
    context = design_request_context(found)
    assert context is not None
    assert json.dumps(context).count("Sign in to continue") == 1

    # A feature that cited nothing is byte-identical to what it was before designs existed.
    await manager.create_technical_prd(feature_id=feature_id, prd=prd)
    assert seen[-1] == [prd]


@pytest.mark.asyncio
async def test_a_frame_too_deep_for_its_budget_renders_shallower_instead_of_vanishing() -> None:
    """The character bound used to be a cliff, and it dropped the frames worth seeing.

    A frame rendered past `MAX_DESIGN_DETAIL_NODE_CHARACTERS` at the configured depth was
    omitted **whole**, so a larger screen reached the Engineer as nothing while a smaller one
    reached it complete -- the worst possible way to spend a budget. Measured on
    AB-Feature-229's login frame, depth 10 renders 276,188 characters and depth 12 renders
    573,536, so raising the depth one step turned a working design into an absent one.

    `_deepest_that_fits` now walks back down until the rendering fits, so the frame arrives at
    the deepest fidelity its budget can pay for. This asserts the *effect* -- a node present,
    rendered shallower -- and not that a helper was called.
    """
    subtree = product_subtree()
    # Both measured the way the resolver renders: a build root with an index tail beneath it.
    deep, _, _ = extract_design_node(
        subtree, depth_bound=MAX_DESIGN_DETAIL_INDEX_DEPTH, build_depth=MAX_DESIGN_DETAIL_DEPTH
    )
    shallow, _, _ = extract_design_node(
        subtree, depth_bound=MAX_DESIGN_DETAIL_INDEX_DEPTH, build_depth=_MIN_DETAIL_DEPTH
    )
    deep_size, shallow_size = _characters(deep), _characters(shallow)
    assert shallow_size < deep_size, "this fixture does not grow with depth; pick another frame"

    client = _ScriptedFigmaClient(
        nodes={PRODUCT_FILE_KEY: product_nodes_result(PRODUCT_FRAME_NODE_ID)}
    )
    # A budget that the deep rendering cannot fit and the shallow one can. Before the fallback
    # this omitted the frame outright.
    detail = await resolver(
        client, node_character_bound=shallow_size + 1, total_character_bound=shallow_size + 1
    ).resolve_detail(
        feature_id="feature-1",
        repository_id="repo",
        workstream_id="repo",
        snapshot_artifact_id="018_design_snapshot.json",
        artifact_id="018_design_detail.repo.json",
        references=[product_citation([PRODUCT_FRAME_NODE_ID])],
        node_ids=[PRODUCT_FRAME_NODE_ID],
    )

    assert detail is not None
    assert not detail.design_nodes_omitted, "the frame was dropped rather than rendered shallower"
    assert len(detail.nodes) == 1
    record = detail.nodes[0]
    assert record.characters <= shallow_size + 1
    # `depth_rendered` is the index tail's depth and no longer varies, so the fallback is
    # asserted on the number it actually moves: how deep build fidelity reached.
    assert record.build_depth_rendered is not None
    assert record.build_depth_rendered < MAX_DESIGN_DETAIL_DEPTH
    # Still a *build* rendering: falling back on depth must never fall back on fidelity, or the
    # Engineer is handed an index record and invents the colours. That is 96-'s Risk 1.
    assert design_content_is_buildable(record.content)


@pytest.mark.asyncio
async def test_a_frame_that_fits_nowhere_is_still_omitted_whole() -> None:
    """The fallback stops at depth 10; it does not trim a frame into a misleading fragment.

    A judge shown two thirds of a frame reads a layout missing children and finds a violation
    that is not there, so an enormous frame is still named as omitted rather than quoted in
    part. What changed is that a frame has to be enormous to earn it.
    """
    client = _ScriptedFigmaClient(
        nodes={PRODUCT_FILE_KEY: product_nodes_result(PRODUCT_FRAME_NODE_ID)}
    )

    detail = await resolver(client, node_character_bound=1, total_character_bound=1).resolve_detail(
        feature_id="feature-1",
        repository_id="repo",
        workstream_id="repo",
        snapshot_artifact_id="018_design_snapshot.json",
        artifact_id="018_design_detail.repo.json",
        references=[product_citation([PRODUCT_FRAME_NODE_ID])],
        node_ids=[PRODUCT_FRAME_NODE_ID],
    )

    assert detail is not None
    assert not detail.nodes
    assert [item.reason for item in detail.design_nodes_omitted] == [
        "over_per_node_character_bound"
    ]


def test_the_hybrid_carries_every_word_the_flat_bound_cut_off() -> None:
    """The measurement that made the hybrid necessary, pinned so it cannot quietly regress.

    AB-Feature-231 produced a screen of correctly-positioned empty boxes. The cause was not the
    model: a flat build rendering bounded at depth 10 carried 24 of the frame's 71 text strings,
    and every string missing from the implementation was a string the platform had withheld.
    "Research & recreate", "Start research", "Generate variations", "Top performing hooks" --
    all of them live at the leaves, below the bound.

    Flat build at depth 14 carries them and costs 754,605 characters, about 188,000 tokens
    against a 272,000-token window, which starves the repository context the change has to be
    written against. The hybrid carries all of them for 433,465 -- inside the per-node bound
    that was already there.
    """
    subtree = product_subtree()

    flat, flat_nodes, _ = extract_design_node(subtree, depth_bound=_MIN_DETAIL_DEPTH)
    hybrid, hybrid_nodes, _ = extract_design_node(
        subtree, depth_bound=MAX_DESIGN_DETAIL_INDEX_DEPTH, build_depth=_MIN_DETAIL_DEPTH
    )

    def strings(content: dict[str, Any]) -> list[str]:
        found: list[str] = []

        def walk(node: Any) -> None:
            if not isinstance(node, dict):
                return
            if node.get("type") == "TEXT" and node.get("text"):
                found.append(str(node["text"]))
            for child in node.get("children") or []:
                walk(child)

        walk(content)
        return found

    flat_text, hybrid_text = strings(flat), strings(hybrid)
    assert len(flat_text) < len(hybrid_text), "the hybrid added no words"
    assert hybrid_nodes > flat_nodes

    # The specific strings whose absence produced the empty boxes.
    for phrase in ("Research & recreate", "Generate variations", "Start research"):
        assert not any(phrase in item for item in flat_text), f"{phrase} was already visible"
        assert any(phrase in item for item in hybrid_text), f"the hybrid still cuts {phrase}"

    # The tail is words only: no paths, no node ids, no style or component names below the
    # build depth. That is what makes it affordable beside the repository snapshot.
    def tail_nodes(content: dict[str, Any], depth: int = 0) -> list[dict[str, Any]]:
        found = [content] if depth > _MIN_DETAIL_DEPTH else []
        for child in content.get("children") or []:
            found.extend(tail_nodes(child, depth + 1))
        return found

    for node in tail_nodes(hybrid):
        assert "path" not in node
        assert "node_id" not in node
        assert "style_names" not in node
        assert set(node) <= {"name", "type", "text", "children", "children_beyond_depth_bound"}

    # Affordable: inside the bound that already existed, and far below flat build fidelity.
    deep, _, _ = extract_design_node(subtree, depth_bound=MAX_DESIGN_DETAIL_DEPTH)
    assert _characters(hybrid) < MAX_DESIGN_DETAIL_NODE_CHARACTERS
    assert _characters(hybrid) < _characters(deep)

    # And still buildable: the tail adds words, it never takes paint off the root. An index
    # record reaching the Engineer whole is 96-'s Risk 1 and this must not become that.
    assert design_content_is_buildable(hybrid)


@pytest.mark.asyncio
async def test_an_image_the_design_fills_a_node_with_is_exported_and_named() -> None:
    """Without this an image is a size and a shrug, and an empty box is the honest answer.

    Figma stores an image behind an `imageRef` no REST caller can resolve to bytes, so the
    rendering carries `{"type": "IMAGE"}` and nothing else. AB-Feature-231 drew exactly that:
    a 862x868 background and five thumbnails, all blank. The node is rendered to PNG through
    the same two calls the console preview already uses.

    Asserted on the rendering as well as the artifact, because the file and the box that needs
    it have to be one fact: a model handed two lists to join by name will not join them.
    """
    stored: list[tuple[str, str, int]] = []

    async def sink(feature_id: str, filename: str, content: bytes) -> str:
        stored.append((feature_id, filename, len(content)))
        return f"attachment-{len(stored)}"

    client = _ScriptedFigmaClient(
        nodes={PRODUCT_FILE_KEY: product_nodes_result(PRODUCT_FRAME_NODE_ID)},
        rendered=b"\x89PNG\r\n\x1a\n" + b"0" * 64,
    )
    detail = await resolver(client, asset_sink=sink).resolve_detail(
        feature_id="feature-1",
        repository_id="repo",
        workstream_id="repo",
        snapshot_artifact_id="018_design_snapshot.json",
        artifact_id="018_design_detail.repo.json",
        references=[product_citation([PRODUCT_FRAME_NODE_ID])],
        node_ids=[PRODUCT_FRAME_NODE_ID],
    )

    assert detail is not None
    assert detail.assets, "the frame's image fills were not exported"
    for asset in detail.assets:
        assert asset.workspace_path.startswith(".design/assets/")
        assert asset.workspace_path.endswith(".png")
        assert asset.attachment_id
        assert asset.bytes_written > 0
    # One store per exported asset, plus one for the frame's own picture -- which is a prompt
    # input for the roles that build and judge, never a file the application uses.
    assert len(stored) == len(detail.assets) + len(detail.nodes)
    assert all(record.preview_attachment_id for record in detail.nodes)
    assert all(
        asset.workspace_path != record.preview_attachment_id
        for asset in detail.assets
        for record in detail.nodes
    ), "the frame picture must never be written into the checkout"

    # The rendering names the file beside the box, so the two are one fact.
    quoted = json.dumps(detail.nodes[0].content)
    for asset in detail.assets:
        assert asset.workspace_path in quoted


@pytest.mark.asyncio
async def test_a_design_with_no_asset_sink_exports_nothing_and_still_resolves() -> None:
    """A deployment without attachment storage keeps exactly the behaviour it had."""
    client = _ScriptedFigmaClient(
        nodes={PRODUCT_FILE_KEY: product_nodes_result(PRODUCT_FRAME_NODE_ID)}
    )
    detail = await resolver(client).resolve_detail(
        feature_id="feature-1",
        repository_id="repo",
        workstream_id="repo",
        snapshot_artifact_id="018_design_snapshot.json",
        artifact_id="018_design_detail.repo.json",
        references=[product_citation([PRODUCT_FRAME_NODE_ID])],
        node_ids=[PRODUCT_FRAME_NODE_ID],
    )

    assert detail is not None
    assert detail.assets == []
    assert detail.assets_omitted == []
    assert "asset_path" not in json.dumps(detail.nodes[0].content)


@pytest.mark.asyncio
async def test_a_render_that_fails_costs_one_image_and_never_the_resolution() -> None:
    """A design with no artwork is still a design worth building against.

    Reported rather than dropped, on the rule every bound here follows: an unreported absence
    reads as "the design had no image there", and the Engineer is then blamed for a box it was
    never given anything to fill.
    """
    refusal = FigmaClientError(
        "refused",
        mode=FigmaFailureMode.REQUEST_REFUSED,
        error_code="figma_render_returned_no_url",
        endpoint="images",
    )

    async def sink(feature_id: str, filename: str, content: bytes) -> str:
        raise AssertionError("nothing should be stored when the render refused")

    client = _ScriptedFigmaClient(
        nodes={PRODUCT_FILE_KEY: product_nodes_result(PRODUCT_FRAME_NODE_ID)},
        render_raises=refusal,
    )
    detail = await resolver(client, asset_sink=sink).resolve_detail(
        feature_id="feature-1",
        repository_id="repo",
        workstream_id="repo",
        snapshot_artifact_id="018_design_snapshot.json",
        artifact_id="018_design_detail.repo.json",
        references=[product_citation([PRODUCT_FRAME_NODE_ID])],
        node_ids=[PRODUCT_FRAME_NODE_ID],
    )

    assert detail is not None
    assert detail.nodes, "the resolution died with the asset"
    assert detail.assets == []
    assert detail.assets_omitted
    assert all(item.reason == "figma_render_returned_no_url" for item in detail.assets_omitted)


@pytest.mark.asyncio
async def test_a_design_asset_is_written_into_the_checkout_and_cannot_escape_it(
    tmp_path: Path,
) -> None:
    """The platform places the bytes, because an agent cannot fetch them: coding tools write text.

    The containment check should never bite -- the path is platform-composed -- which is the
    reason to have it rather than the reason to skip it. A design artifact is durable state,
    and durable state that names `../../etc` has to be refused where it is used, not where it
    was written.
    """
    from services.feature_runtime import _place_design_assets

    class _Bytes:
        async def get_content(self, attachment_id: str) -> bytes | None:
            return b"\x89PNG\r\n\x1a\n" + attachment_id.encode()

    workspace = tmp_path / "checkout"
    workspace.mkdir()
    detail = DesignDetailArtifact.model_validate(
        {
            "schema_version": "1.0.0",
            "workflow_id": "feature-1",
            "artifact_id": "018_design_detail.repo.json",
            "producer": "design_resolver",
            "timestamp": datetime(2026, 9, 13, tzinfo=UTC),
            "metadata": {},
            "validation_status": "valid",
            "feature_id": "feature-1",
            "repository_id": "repo",
            "workstream_id": "repo",
            "snapshot_artifact_id": "018_design_snapshot.json",
            "resolved_at": datetime(2026, 9, 13, tzinfo=UTC),
            "assets": [
                {
                    "node_id": "1:1",
                    "workspace_path": ".design/assets/1-1-image.png",
                    "attachment_id": "a1",
                },
                {
                    "node_id": "2:2",
                    "workspace_path": "../escaped.png",
                    "attachment_id": "a2",
                },
            ],
        }
    )

    await _place_design_assets(
        workspace,
        detail,
        source=cast(Any, _Bytes()),
        runner=cast(Any, _NoGit()),
        timeout=5.0,
        cancellation_token=MockCancellationToken(),
    )

    placed = workspace / ".design/assets/1-1-image.png"
    assert placed.is_file()
    assert placed.read_bytes().startswith(b"\x89PNG")
    assert not (tmp_path / "escaped.png").exists(), "an asset escaped the workspace"


@pytest.mark.asyncio
async def test_placing_assets_without_a_content_source_writes_nothing(tmp_path: Path) -> None:
    """A composition with no attachment storage is exactly what it was before assets existed."""
    from services.feature_runtime import _place_design_assets

    workspace = tmp_path / "checkout"
    workspace.mkdir()
    detail = DesignDetailArtifact.model_validate(
        {
            "schema_version": "1.0.0",
            "workflow_id": "feature-1",
            "artifact_id": "018_design_detail.repo.json",
            "producer": "design_resolver",
            "timestamp": datetime(2026, 9, 13, tzinfo=UTC),
            "metadata": {},
            "validation_status": "valid",
            "feature_id": "feature-1",
            "repository_id": "repo",
            "workstream_id": "repo",
            "snapshot_artifact_id": "018_design_snapshot.json",
            "resolved_at": datetime(2026, 9, 13, tzinfo=UTC),
            "assets": [
                {"node_id": "1:1", "workspace_path": ".design/assets/a.png", "attachment_id": "a1"}
            ],
        }
    )

    await _place_design_assets(
        workspace,
        detail,
        source=None,
        runner=cast(Any, _NoGit()),
        timeout=5.0,
        cancellation_token=MockCancellationToken(),
    )

    assert not (workspace / ".design").exists()


@pytest.mark.asyncio
async def test_a_coding_call_shows_the_design_only_to_a_model_that_can_see() -> None:
    """Images are dropped for a blind model rather than sent and refused.

    A provider error on a coding call costs the attempt, and a coding call that reads the
    design as text is exactly what this platform did until now -- so the degraded path has to
    be the old path, not a failure.
    """
    from adapters.llm_adapter import ImageInput, LLMResponse, ResponsesCodingExecutor

    class _Client:
        def __init__(self, *, sees: bool) -> None:
            self.vision_capable = sees
            self.images_seen: list[int] = []

        async def respond(self, *, instructions: str, input_text: str, images: Any = ()) -> Any:
            del instructions, input_text
            self.images_seen.append(len(images))
            return LLMResponse(
                output_text='{"summary": "s", "files": []}',
                response_id="r",
                model="m",
                input_tokens=1,
                output_tokens=1,
            )

    picture = (ImageInput(media_type="image/png", data="Zm9v"),)

    for sees, expected in ((True, 1), (False, 0)):
        client = _Client(sees=sees)
        executor = ResponsesCodingExecutor(llm_client=cast(Any, client))
        with suppress(Exception):
            await executor.execute(
                workspace_root=Path("."),
                instructions="do it",
                input_text="{}",
                images=picture,
            )
        assert client.images_seen == [expected], (
            "a blind model was sent an image" if not sees else "a seeing model was shown none"
        )


@pytest.mark.asyncio
async def test_assets_are_never_placed_into_a_directory_the_clone_still_needs(
    tmp_path: Path,
) -> None:
    """Writing artwork before the checkout exists destroys the checkout.

    `mkdir(parents=True)` on `<workspace>/.design/assets` creates the workspace directory, and
    the clone then refuses it with "workflow workspace already exists". AB-Feature-233 died
    exactly there, before its first attempt, having successfully exported all six images --
    the export worked and the placement killed the run.

    Asserted as the property that matters: placing assets must never bring a non-existent
    workspace into existence.
    """
    from services.feature_runtime import _place_design_assets

    class _Bytes:
        async def get_content(self, attachment_id: str) -> bytes | None:
            return b"\x89PNG\r\n\x1a\n"

    missing = tmp_path / "not-cloned-yet"
    detail = DesignDetailArtifact.model_validate(
        {
            "schema_version": "1.0.0",
            "workflow_id": "feature-1",
            "artifact_id": "018_design_detail.repo.json",
            "producer": "design_resolver",
            "timestamp": datetime(2026, 9, 13, tzinfo=UTC),
            "metadata": {},
            "validation_status": "valid",
            "feature_id": "feature-1",
            "repository_id": "repo",
            "workstream_id": "repo",
            "snapshot_artifact_id": "018_design_snapshot.json",
            "resolved_at": datetime(2026, 9, 13, tzinfo=UTC),
            "assets": [
                {
                    "node_id": "1:1",
                    "workspace_path": ".design/assets/a.png",
                    "attachment_id": "a1",
                }
            ],
        }
    )

    await _place_design_assets(
        missing,
        detail,
        source=cast(Any, _Bytes()),
        runner=cast(Any, _NoGit()),
        timeout=5.0,
        cancellation_token=MockCancellationToken(),
    )

    assert not missing.exists(), (
        "placing assets created the workspace directory; the clone will refuse it"
    )


class _NoGit:
    """A runner that lists no tracked files, so placement keeps the artifact's own path."""

    async def run(
        self,
        command: Any,
        cwd: Any,
        timeout_seconds: float,
        cancellation_token: Any,
        environment: Any = None,
    ) -> Any:
        """Answer an empty `git ls-files`."""
        del command, cwd, timeout_seconds, cancellation_token, environment
        return SimpleNamespace(return_code=0, stdout="", stderr="")


@pytest.mark.asyncio
async def test_artwork_is_placed_where_the_repository_already_keeps_images(
    tmp_path: Path,
) -> None:
    """The platform decides the directory, because the Engineer cannot.

    Its tools write text, so it cannot move a PNG. AB-Feature-236 was told to "move each file
    to wherever this repository keeps static assets", correctly referenced
    `/74046-28182-image-41.png`, and could not move the file -- so the reference pointed at
    nothing.

    The directory is read off the checkout's own tracked files rather than assumed from a
    framework convention: whichever directory already holds the most committed images is where
    the next image belongs.
    """
    from services.feature_runtime import _place_design_assets

    class _Bytes:
        async def get_content(self, attachment_id: str) -> bytes | None:
            return b"\x89PNG\r\n\x1a\n"

    class _Tracked:
        async def run(
            self,
            command: Any,
            cwd: Any,
            timeout_seconds: float,
            cancellation_token: Any,
            environment: Any = None,
        ) -> Any:
            del command, cwd, timeout_seconds, cancellation_token, environment
            listing = "\n".join(
                ["public/logo.svg", "public/hero.png", "public/icon.svg", "src/app/page.tsx"]
            )
            return SimpleNamespace(return_code=0, stdout=listing, stderr="")

    workspace = tmp_path / "checkout"
    workspace.mkdir()
    detail = DesignDetailArtifact.model_validate(
        {
            "schema_version": "1.0.0",
            "workflow_id": "feature-1",
            "artifact_id": "018_design_detail.repo.json",
            "producer": "design_resolver",
            "timestamp": datetime(2026, 9, 13, tzinfo=UTC),
            "metadata": {},
            "validation_status": "valid",
            "feature_id": "feature-1",
            "repository_id": "repo",
            "workstream_id": "repo",
            "snapshot_artifact_id": "018_design_snapshot.json",
            "resolved_at": datetime(2026, 9, 13, tzinfo=UTC),
            "assets": [
                {
                    "node_id": "1:1",
                    "workspace_path": ".design/assets/74046-28182-image-41.png",
                    "attachment_id": "a1",
                }
            ],
        }
    )

    await _place_design_assets(
        workspace,
        detail,
        source=cast(Any, _Bytes()),
        runner=cast(Any, _Tracked()),
        timeout=5.0,
        cancellation_token=MockCancellationToken(),
    )

    # Placed where this repository keeps images, under its own filename.
    assert (workspace / "public/74046-28182-image-41.png").is_file()
    # And not left in the platform's staging path, which nothing serves.
    assert not (workspace / ".design/assets/74046-28182-image-41.png").exists()


@pytest.mark.asyncio
async def test_placement_detects_the_directory_even_with_no_runner_supplied(
    tmp_path: Path,
) -> None:
    """An absent runner means "build the default", never "skip the detection".

    That is this codebase's convention -- `ReviewerAgent` and both in-attempt checkers do
    `process_runner or AsyncioProcessRunner(...)` -- and `LiveChildWorkstreamExecutor` is
    constructed without one. Reading `None` as "do not detect" silently sent every exported
    image to `.design/assets`, which nothing serves: AB-Feature-237 referenced six public
    URLs that resolved to nothing, and its own self-review said exactly that.
    """
    from services.feature_runtime import _place_design_assets

    class _Bytes:
        async def get_content(self, attachment_id: str) -> bytes | None:
            return b"\x89PNG\r\n\x1a\n"

    workspace = tmp_path / "checkout"
    (workspace / "images").mkdir(parents=True)
    subprocess.run(["git", "init", "--quiet"], cwd=workspace, check=True)
    for name in ("a.png", "b.png", "c.svg"):
        (workspace / "images" / name).write_bytes(b"\x89PNG\r\n\x1a\n")
    subprocess.run(["git", "add", "-A"], cwd=workspace, check=True)

    detail = DesignDetailArtifact.model_validate(
        {
            "schema_version": "1.0.0",
            "workflow_id": "feature-1",
            "artifact_id": "018_design_detail.repo.json",
            "producer": "design_resolver",
            "timestamp": datetime(2026, 9, 13, tzinfo=UTC),
            "metadata": {},
            "validation_status": "valid",
            "feature_id": "feature-1",
            "repository_id": "repo",
            "workstream_id": "repo",
            "snapshot_artifact_id": "018_design_snapshot.json",
            "resolved_at": datetime(2026, 9, 13, tzinfo=UTC),
            "assets": [
                {
                    "node_id": "1:1",
                    "workspace_path": ".design/assets/hero.png",
                    "attachment_id": "a1",
                }
            ],
        }
    )

    # No runner at all -- the call site does not have one to give.
    await _place_design_assets(
        workspace,
        detail,
        source=cast(Any, _Bytes()),
        runner=None,
        timeout=30.0,
        cancellation_token=MockCancellationToken(),
    )

    assert (workspace / "images/hero.png").is_file(), "the detection was skipped"
    assert not (workspace / ".design/assets/hero.png").exists()


@pytest.mark.asyncio
async def test_placed_artwork_is_committed_because_only_the_platform_can_commit_it(
    tmp_path: Path,
) -> None:
    """A file in the workspace that never reaches the branch is a 404 in the pull request.

    The attempt's own commit stages exactly the reviewed paths and is bound to their content
    fingerprint -- a boundary that correctly refuses these binaries, because the Engineer did
    not write them and the Reviewer did not review their bytes. So nothing committed them.

    AB-Feature-237 raised a pull request referencing `/74046-28182-image-41.png` and five
    others, all of which 404 on the branch: artwork in the workspace, code pointing at it
    correctly, and no commit anywhere.
    """
    from services.feature_runtime import _place_design_assets

    class _Bytes:
        async def get_content(self, attachment_id: str) -> bytes | None:
            return b"\x89PNG\r\n\x1a\n" + attachment_id.encode()

    workspace = tmp_path / "checkout"
    (workspace / "public").mkdir(parents=True)
    subprocess.run(["git", "init", "--quiet"], cwd=workspace, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=workspace, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=workspace, check=True)
    (workspace / "public" / "existing.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    subprocess.run(["git", "add", "-A"], cwd=workspace, check=True)
    subprocess.run(["git", "commit", "--quiet", "-m", "base"], cwd=workspace, check=True)

    detail = DesignDetailArtifact.model_validate(
        {
            "schema_version": "1.0.0",
            "workflow_id": "feature-1",
            "artifact_id": "018_design_detail.repo.json",
            "producer": "design_resolver",
            "timestamp": datetime(2026, 9, 13, tzinfo=UTC),
            "metadata": {},
            "validation_status": "valid",
            "feature_id": "feature-1",
            "repository_id": "repo",
            "workstream_id": "repo",
            "snapshot_artifact_id": "018_design_snapshot.json",
            "resolved_at": datetime(2026, 9, 13, tzinfo=UTC),
            "assets": [
                {
                    "node_id": "1:1",
                    "workspace_path": ".design/assets/hero.png",
                    "attachment_id": "a1",
                }
            ],
        }
    )

    await _place_design_assets(
        workspace,
        detail,
        source=cast(Any, _Bytes()),
        runner=None,
        timeout=30.0,
        cancellation_token=MockCancellationToken(),
    )

    tracked = subprocess.run(
        ["git", "ls-files", "public/hero.png"],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert tracked.strip() == "public/hero.png", "the artwork never reached the branch"

    # Committed, not merely staged: a staged-but-uncommitted tree would also make the
    # attempt's own commit refuse with "pre-existing staged changes".
    staged = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert staged.strip() == "", "the index was left dirty for the attempt's commit to trip on"
