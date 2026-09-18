"""Turn the designs somebody cited into one artifact, once, and never re-read them.

A citation is resolved exactly once. Attempt 2 of a workstream builds against the same
snapshot attempt 1 did, and a review judges against the snapshot the attempt was given.
Re-fetching per attempt would mean a design edited mid-feature silently changes the work order
and the resulting diff is unexplainable -- which is the failure 67- was about, in a new place.
A refresh is a **new revision** of the artifact, never an edit of one.

**Names outrank values.** A hex code tells an engineer what to hardcode; a style name
(`color/surface/raised`, `Button/Primary/Hover`) tells it what the repository already has and
what it must reuse -- which is the difference between a change that matches the design system
and one that matches the picture. So every rendered node puts its `style_names` before its
resolved fills, and an instance names its component and its component set before its geometry.

**Bounds report what they dropped.** Three lists, and each is populated independently:

* `design_nodes_omitted` -- cited nodes that resolved but did not fit, with the size measured
  and the bound that bit. Omitted **whole**, never trimmed: a judge shown two thirds of a
  frame reads a layout that is missing children and finds a violation that is not there.
* `design_nodes_absent` -- cited nodes the file does not define. Figma reports this by
  answering `200` with `nodes[id] = null`, which is an answer and not a failure.
* `design_nodes_unreachable` -- cited nodes this deployment's token could not read.

An unreported cap reads as "the whole design was considered", so none of them is silent and
all three reach the prompt and the console as first-class content.

**Style names, not variables.** Measured against the live API on 2026-09-07:
`GET /v1/files/:key/variables/local` answers `403 Invalid scope(s) ... requires the
file_variables:read scope` for a read-only personal access token, and parts of that API are
Enterprise-only. So the resolution reads the style names already present on the nodes, and the
snapshot records `style_name_source` so a reader knows which of the two it is looking at rather
than assuming the richer one.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, Protocol, cast

from adapters.figma_adapter import (
    FigmaClientError,
    FigmaDesignClient,
    FigmaNodeSubtree,
)
from agents.shared.contracts import FEATURE_ARTIFACT_FILENAMES, create_artifact
from artifacts.design_references import (
    DESIGN_BUILD_ONLY_KEYS,
    DESIGN_INDEX_KEYS,
    MAX_DESIGN_DEPTH,
    MAX_DESIGN_DETAIL_DEPTH,
    MAX_DESIGN_DETAIL_INDEX_DEPTH,
    MAX_DESIGN_DETAIL_NODE_CHARACTERS,
    MAX_DESIGN_DETAIL_TOTAL_CHARACTERS,
    MAX_DESIGN_INDEX_NODE_CHARACTERS,
    MAX_DESIGN_INDEX_TOTAL_CHARACTERS,
    MAX_DESIGN_NODES,
    DesignReference,
    DesignRenderMode,
    design_content_is_buildable,
)
from artifacts.schemas import (
    DesignAsset,
    DesignAssetOmission,
    DesignDetailArtifact,
    DesignNodeCitation,
    DesignNodeOmission,
    DesignNodeRecord,
    DesignSnapshotArtifact,
    DesignSourceFile,
)
from state.failure_diagnosis import FeatureFailureClassification

# The style-name provenance values. `node_styles` is what a read-only personal access token can
# reach; `file_variables` is what an Enterprise deployment holding `file_variables:read` could
# reach, and nothing in this platform asks for it today. Recorded rather than assumed, so a
# reader is never left thinking the richer source was used.
STYLE_NAMES_FROM_NODES = "node_styles"

# The auto-layout facts a person writing a component actually needs. Read as *declared*: Figma
# omits a zero padding and a zero item spacing entirely, so an absent key means the default and
# not "unknown" -- which is why they are copied through rather than defaulted to numbers here.
_LAYOUT_KEYS = (
    "layoutMode",
    "layoutWrap",
    "primaryAxisAlignItems",
    "counterAxisAlignItems",
    "primaryAxisSizingMode",
    "counterAxisSizingMode",
    "layoutSizingHorizontal",
    "layoutSizingVertical",
    "layoutGrow",
    "layoutAlign",
    "itemSpacing",
    "counterAxisSpacing",
    "paddingLeft",
    "paddingRight",
    "paddingTop",
    "paddingBottom",
)

# The typography facts, and the Figma key each is read from.
_TYPOGRAPHY_KEYS = (
    ("family", "fontFamily"),
    ("size", "fontSize"),
    ("weight", "fontWeight"),
    ("style", "fontStyle"),
    ("line_height", "lineHeightPx"),
    ("letter_spacing", "letterSpacing"),
    ("case", "textCase"),
    ("decoration", "textDecoration"),
    ("align", "textAlignHorizontal"),
)

_COMPONENT_NODE_TYPES = frozenset({"INSTANCE", "COMPONENT", "COMPONENT_SET"})

# The shallowest rendering `_deepest_that_fits` will fall back to, and the step it falls by.
# `_MIN_DETAIL_DEPTH` is the depth this platform shipped before the fallback existed, so a
# frame that fits today still renders exactly as it does today; the fallback only ever adds
# frames that used to be omitted outright.
_MIN_DETAIL_DEPTH = 10
_DETAIL_DEPTH_STEP = 2

# How many images one workstream's design may place. Six covers the frame measured (a
# background and five thumbnails) and stays inside the attachment store's per-feature count,
# which a submission's own images also spend. Over it is reported, never silently dropped.
MAX_DESIGN_ASSETS = 6

# Where exported images are written, relative to the repository root. Deliberately neutral:
# this platform does not know whether a repository serves static files from `public/`,
# `static/`, `assets/` or an import pipeline, and encoding one would be exactly the
# target-repository assumption it must not make. The Engineer is told the file is here and
# told to move it where *this* repository keeps such things.
_ASSET_DIRECTORY = ".design/assets"

_ASSET_OVER_COUNT_BOUND = "over_asset_count_bound"
_ASSET_VERSION_UNKNOWN = "file_version_not_recorded"

_OMITTED_OVER_NODE_BOUND = "over_per_node_character_bound"
_OMITTED_OVER_TOTAL_BOUND = "over_total_character_bound"
_OMITTED_OVER_NODE_COUNT_BOUND = "over_node_count_bound"

# The index tier's own two reasons, kept distinct from the detail tier's above so a reader can
# tell "this flow is too large to even list" from "this workstream already has enough screens".
# Both should be unreachable on any real design -- see `_index_omission_reason`.
_OMITTED_OVER_INDEX_NODE_BOUND = "over_index_per_node_character_bound"
_OMITTED_INDEX_TOTAL_EXHAUSTED = "index_total_exhausted"

_ABSENT_NOT_IN_FILE = "not_defined_in_file"
_UNREACHABLE_TOKEN_REFUSED = "token_could_not_read_the_file"


class DesignResolutionRefused(Exception):
    """Raised when a citation cannot be resolved and nobody should build against prose.

    Deliberately not a fault. The provider answered -- a wrong URL, a frame that was deleted, a
    file this deployment's token cannot open -- so retrying re-asks a question whose answer is
    already known. It stops the feature **before planning**, which costs nothing but the
    person's correction: a refusal before any effect is not a failure (70-), and no repository
    has been touched at the point this is raised.

    ``diagnostics`` is written here from the platform's own words and the file key the person
    pasted -- never a provider message.

    It declares its classification the way the adapters do, because the generic terminal
    handler reads it off the exception: without one, ``normalize_classification`` files the
    type name as ``PLATFORM_DEFECT`` -- sending somebody to debug this codebase over a URL
    only its author can correct. ``FEATURE_QUEUE_REFUSED`` is the value the platform already
    uses for a refusal a person can fix (`fail_queued`'s default), and it is not retryable,
    which is right: the provider answered, and answers the same way every time.
    """

    failure_classification = FeatureFailureClassification.FEATURE_QUEUE_REFUSED

    def __init__(self, message: str, *, diagnostics: Sequence[str] = ()) -> None:
        """Keep the actionable sentence available to the control plane."""
        self.diagnostics = tuple(diagnostics)
        super().__init__(message)


class DesignAssetSink(Protocol):
    """Store one exported design image durably, and answer with its identifier.

    A protocol rather than the store itself so the resolver never learns where bytes live: a
    deployment with no attachment storage supplies nothing, exports nothing, and reports every
    image as unplaced -- which is the behaviour this platform had before assets existed.
    """

    async def __call__(self, feature_id: str, filename: str, content: bytes) -> str:
        """Persist the bytes against this feature and return an identifier, or an empty one."""


class DesignReferenceResolver(Protocol):
    """Turn a feature's citations into one design snapshot artifact, and its frames into detail.

    Both halves are on one protocol because both resolvers must implement both. Every test
    outside the live tier runs `DeterministicDesignResolver`, so a method only the Figma
    resolver has is a method nothing fails on until a real feature runs -- which is Risk 4 of
    96- and the reason `resolve_detail` is declared here rather than only where it is used.
    """

    async def resolve(
        self,
        *,
        feature_id: str,
        references: Sequence[DesignReference],
        artifact_id: str,
    ) -> DesignSnapshotArtifact:
        """Resolve every citation into one artifact, or refuse the whole resolution."""

    async def resolve_detail(
        self,
        *,
        feature_id: str,
        repository_id: str,
        workstream_id: str,
        snapshot_artifact_id: str,
        references: Sequence[DesignReference],
        node_ids: Sequence[str],
        artifact_id: str,
    ) -> DesignDetailArtifact:
        """Resolve one repository's assigned frames at build fidelity.

        `node_ids` is the workstream's own assignment in plan order, and the bound is spent in
        that order -- so a frame the plan named first is the frame that fits. Never refuses the
        way `resolve` does: a workstream whose frames could not be read is told so through the
        omission lists and builds from prose, because there is no fault allowance at this seam
        and killing the workstream would cost more than an honest gap.
        """


class DeterministicDesignResolver:
    """Resolve citations to a fixed snapshot shape with no network call and no adapter.

    Mock mode runs a deterministic orchestrator with no adapters and no LLM client, and it
    exists so the *sequence* can be exercised without any provider existing. This is the
    design-resolution half of that, beside `DeterministicFeatureProductManager`: a citation on
    a mock feature produces the same artifact contract, no `fetch_design_reference` row, and no
    outbound call -- so the step's place in the order is covered by the mock tier.

    A mock deployment with no design source configured still refuses citations at submission
    exactly as a live one does; this serves the deployment that has one.
    """

    async def resolve(
        self,
        *,
        feature_id: str,
        references: Sequence[DesignReference],
        artifact_id: str,
    ) -> DesignSnapshotArtifact:
        """Produce one snapshot whose content is derived from the citation and nothing else."""
        nodes: list[DesignNodeRecord] = []
        files: dict[str, DesignSourceFile] = {}
        spent = 0
        for reference in references:
            files.setdefault(
                reference.file_key,
                DesignSourceFile(
                    file_key=reference.file_key,
                    file_name=f"mock file {reference.file_key}",
                    # Fixed rather than absent: the preview endpoint pins to whatever the
                    # snapshot recorded, and a mock snapshot recording nothing would make that
                    # path untestable.
                    file_version="mock-version",
                ),
            )
            # A whole-file citation resolves to one deterministic stand-in frame, because a
            # mock has no file to list.
            for node_id in reference.node_ids or ["0:1"]:
                # Index-shaped, like the live snapshot: names, structure and text, and no
                # `layout`. A mock snapshot carrying build-fidelity keys would make the mock
                # tier the one place an index record looks buildable, which is precisely the
                # confusion `design_content_is_buildable` exists to prevent.
                content = {
                    "path": f"mock/{reference.label or node_id}",
                    "name": reference.label or f"Frame {node_id}",
                    "type": "FRAME",
                    "style_names": {"fills": "mock/surface"},
                    "children": [],
                }
                characters = _characters(content)
                spent += characters
                nodes.append(
                    DesignNodeRecord(
                        file_key=reference.file_key,
                        node_id=node_id,
                        label=reference.label,
                        source_url=reference.url,
                        applies_to=list(reference.applies_to),
                        content=content,
                        characters=characters,
                        nodes_rendered=1,
                        depth_rendered=0,
                    )
                )
        return _snapshot(
            feature_id=feature_id,
            artifact_id=artifact_id,
            files=list(files.values()),
            nodes=nodes,
            omitted=[],
            absent=[],
            unreachable=[],
            characters_selected=spent,
            resolution_mode="mock",
        )

    async def resolve_detail(
        self,
        *,
        feature_id: str,
        repository_id: str,
        workstream_id: str,
        snapshot_artifact_id: str,
        references: Sequence[DesignReference],
        node_ids: Sequence[str],
        artifact_id: str,
    ) -> DesignDetailArtifact:
        """Produce one repository's detail with no network call, in the live artifact's shape.

        Implemented here and not only on the Figma resolver, deliberately: every test outside
        the live tier runs this class, so a `resolve_detail` only the real resolver had would
        pass the whole mock tier and fail nothing until a live feature ran (Risk 4 of 96-).

        The content carries `layout` as well as `style_names`, unlike the index records the
        mock snapshot writes, because the point of this tier is that it is buildable -- and
        `design_content_is_buildable` is asserted against exactly this shape.
        """
        provenance = _provenance_by_node(references)
        nodes: list[DesignNodeRecord] = []
        files: dict[str, DesignSourceFile] = {}
        spent = 0
        for node_id in dict.fromkeys(node_ids):
            citation = provenance.get(node_id)
            if citation is None:
                continue
            files.setdefault(
                citation.file_key,
                DesignSourceFile(
                    file_key=citation.file_key,
                    file_name=f"mock file {citation.file_key}",
                    file_version="mock-version",
                ),
            )
            content = {
                "path": f"mock/{citation.label or node_id}",
                "name": citation.label or f"Frame {node_id}",
                "type": "FRAME",
                "style_names": {"fills": "mock/surface"},
                "layout": {"layoutMode": "VERTICAL", "itemSpacing": 8},
                "size": {"width": 360.0, "height": 640.0},
                "children": [],
            }
            characters = _characters(content)
            spent += characters
            nodes.append(
                DesignNodeRecord(
                    file_key=citation.file_key,
                    node_id=node_id,
                    label=citation.label,
                    source_url=citation.source_url,
                    applies_to=list(citation.applies_to),
                    content=content,
                    characters=characters,
                    nodes_rendered=1,
                    depth_rendered=0,
                )
            )
        return _detail(
            feature_id=feature_id,
            repository_id=repository_id,
            workstream_id=workstream_id,
            snapshot_artifact_id=snapshot_artifact_id,
            artifact_id=artifact_id,
            files=list(files.values()),
            nodes=nodes,
            omitted=[],
            absent=[],
            unreachable=[],
            characters_selected=spent,
            resolution_mode="mock",
        )


class FigmaDesignResolver:
    """Resolve citations by reading the cited nodes once, through the design adapter.

    Holds no token. The client is built per resolution by the factory it was given, which is
    where the configured owner's `figma` credential is opened -- so the token never enters a
    credentials record that is passed around, never reaches an artifact, a log, a prompt or the
    client, and exists only for the duration of the calls that need it.
    """

    def __init__(
        self,
        *,
        client_factory: Callable[[], Awaitable[FigmaDesignClient | None]],
        node_bound: int = MAX_DESIGN_NODES,
        index_node_character_bound: int = MAX_DESIGN_INDEX_NODE_CHARACTERS,
        index_total_character_bound: int = MAX_DESIGN_INDEX_TOTAL_CHARACTERS,
        index_depth_bound: int = MAX_DESIGN_DEPTH,
        node_character_bound: int = MAX_DESIGN_DETAIL_NODE_CHARACTERS,
        total_character_bound: int = MAX_DESIGN_DETAIL_TOTAL_CHARACTERS,
        depth_bound: int = MAX_DESIGN_DETAIL_DEPTH,
        index_tail_depth: int = MAX_DESIGN_DETAIL_INDEX_DEPTH,
        asset_sink: DesignAssetSink | None = None,
    ) -> None:
        """Bind the client factory and both tiers' bounds, calibrated by default.

        Two sets, because one resolver answers two different questions. `index_*` bounds the
        snapshot -- every cited frame, rendered for deciding -- and the unprefixed ones bound
        the per-repository detail, which is what an engineer builds from. The unprefixed names
        are the ones that were there before 96-, kept so every existing caller that overrides a
        bound is still overriding the one an engineer is held to.
        """
        self._client_factory = client_factory
        self._node_bound = node_bound
        self._index_node_character_bound = index_node_character_bound
        self._index_total_character_bound = index_total_character_bound
        self._index_depth_bound = index_depth_bound
        self._node_character_bound = node_character_bound
        self._total_character_bound = total_character_bound
        self._depth_bound = depth_bound
        self._index_tail_depth = index_tail_depth
        self._asset_sink = asset_sink

    async def resolve(
        self,
        *,
        feature_id: str,
        references: Sequence[DesignReference],
        artifact_id: str,
    ) -> DesignSnapshotArtifact:
        """Read every citation once and render one bounded, honest index of it.

        The **index**: every cited frame as structure, names and text. It is what the product
        manager derives requirements from and what the planner assigns frames from, and it is
        deliberately complete -- a frame too large to *build* is still a frame the flow has, and
        omitting it here would hide it from the role that decides which repository it belongs
        to. That is the failure 96- exists to fix.

        What an engineer builds from is resolved separately, per repository, once that
        repository's workstream starts. See `resolve_detail`.
        """
        client = await self._client_factory()
        if client is None:
            msg = "this deployment cannot open a design file"
            raise DesignResolutionRefused(
                msg,
                diagnostics=[
                    "This feature cites a design, but no Figma credential could be opened for "
                    "the configured design source. Store one in Settings and resume."
                ],
            )
        citations = await self._citations_to_read(client, references)
        nodes: list[DesignNodeRecord] = []
        omitted: list[DesignNodeOmission] = []
        absent: list[DesignNodeCitation] = []
        unreachable: list[DesignNodeCitation] = []
        files: dict[str, DesignSourceFile] = {}
        spent = 0
        for file_key, wanted in _by_file(citations).items():
            try:
                result = await client.fetch_nodes(file_key, [item.node_id for item in wanted])
            except FigmaClientError as error:
                if not error.is_the_persons_to_fix:
                    # The provider did not answer. Raised so the step's own fault allowance can
                    # spend a retry on it; a provider fault when it stays unanswered.
                    raise
                unreachable.extend(
                    DesignNodeCitation(
                        file_key=file_key,
                        node_id=item.node_id,
                        label=item.label,
                        source_url=item.source_url,
                        reason=_UNREACHABLE_TOKEN_REFUSED,
                    )
                    for item in wanted
                )
                continue
            files[file_key] = DesignSourceFile(
                file_key=file_key,
                file_name=result.file_name,
                file_version=result.file_version,
            )
            by_id = {item.node_id: item for item in wanted}
            for node_id in result.absent_node_ids:
                citation = by_id[node_id]
                absent.append(
                    DesignNodeCitation(
                        file_key=file_key,
                        node_id=node_id,
                        label=citation.label,
                        source_url=citation.source_url,
                        reason=_ABSENT_NOT_IN_FILE,
                    )
                )
            for subtree in result.subtrees:
                citation = by_id[subtree.node_id]
                if len(nodes) >= self._node_bound:
                    omitted.append(
                        DesignNodeOmission(
                            file_key=file_key,
                            node_id=subtree.node_id,
                            label=citation.label,
                            reason=_OMITTED_OVER_NODE_COUNT_BOUND,
                            characters=0,
                        )
                    )
                    continue
                content, rendered, depth = extract_design_node(
                    subtree, depth_bound=self._index_depth_bound, detail="index"
                )
                characters = _characters(content)
                reason = self._index_omission_reason(characters, spent=spent)
                if reason is not None:
                    omitted.append(
                        DesignNodeOmission(
                            file_key=file_key,
                            node_id=subtree.node_id,
                            label=citation.label,
                            reason=reason,
                            characters=characters,
                        )
                    )
                    continue
                spent += characters
                nodes.append(
                    DesignNodeRecord(
                        file_key=file_key,
                        node_id=subtree.node_id,
                        label=citation.label,
                        source_url=citation.source_url,
                        applies_to=list(citation.applies_to),
                        content=content,
                        characters=characters,
                        nodes_rendered=rendered,
                        depth_rendered=depth,
                    )
                )
        if not nodes:
            # Nothing resolved at all, so this feature would be planned, built and reviewed
            # against prose with a mock attached -- the class of lie 77- exists to stop. The
            # partial case is deliberately *not* this: a snapshot holding two of three frames
            # is useful, and the third is named in a list every judge is shown.
            raise DesignResolutionRefused(
                "no cited design could be resolved",
                diagnostics=[_no_nodes_diagnostic(absent, unreachable)],
            )
        return _snapshot(
            feature_id=feature_id,
            artifact_id=artifact_id,
            files=list(files.values()),
            nodes=nodes,
            omitted=omitted,
            absent=absent,
            unreachable=unreachable,
            characters_selected=spent,
            resolution_mode="live",
            node_bound=self._node_bound,
            node_character_bound=self._index_node_character_bound,
            total_character_bound=self._index_total_character_bound,
            depth_bound=self._index_depth_bound,
        )

    async def resolve_detail(
        self,
        *,
        feature_id: str,
        repository_id: str,
        workstream_id: str,
        snapshot_artifact_id: str,
        references: Sequence[DesignReference],
        node_ids: Sequence[str],
        artifact_id: str,
    ) -> DesignDetailArtifact:
        """Read this repository's assigned frames once, at build fidelity, and report the rest.

        Deliberately unlike `resolve` in one respect: it never raises `DesignResolutionRefused`
        when nothing resolves. `resolve` runs before planning, where refusing costs a person a
        correction and no repository has been touched; this runs when a workstream is starting,
        where refusing would end the workstream over a design it can be *told* it did not get.
        Every failure here lands in an omission list the prompts already know how to read.

        A `FigmaClientError` is not caught here at all -- the caller owns that, because the
        remedy differs by seam. See `_resolved_design_detail` in `feature_workflow.py`: it
        spends a small local allowance on a transient fault and then degrades to "unreachable"
        rather than killing the workstream, which is what propagating out of the child loop's
        pre-loop section would do.
        """
        wanted = list(dict.fromkeys(node_ids))
        provenance = _provenance_by_node(references)
        nodes: list[DesignNodeRecord] = []
        omitted: list[DesignNodeOmission] = []
        absent: list[DesignNodeCitation] = []
        unreachable: list[DesignNodeCitation] = []
        files: dict[str, DesignSourceFile] = {}
        spent = 0
        client = await self._client_factory()
        if client is None:
            # No credential could be opened. The snapshot already exists, so the feature is
            # past the point where refusing is cheap: every assigned frame is reported
            # unreachable and the workstream builds knowing it was not shown them.
            return _detail(
                feature_id=feature_id,
                repository_id=repository_id,
                workstream_id=workstream_id,
                snapshot_artifact_id=snapshot_artifact_id,
                artifact_id=artifact_id,
                files=[],
                nodes=[],
                omitted=[],
                absent=[],
                unreachable=[_unreachable_citation(node_id, provenance) for node_id in wanted],
                characters_selected=0,
                resolution_mode="live",
                node_character_bound=self._node_character_bound,
                total_character_bound=self._total_character_bound,
                depth_bound=self._depth_bound,
            )
        for file_key, group in _by_file(
            [provenance[node_id] for node_id in wanted if node_id in provenance]
        ).items():
            try:
                result = await client.fetch_nodes(file_key, [item.node_id for item in group])
            except FigmaClientError as error:
                if not error.is_the_persons_to_fix:
                    # The provider did not answer. Raised for the caller's own allowance, which
                    # is where the retry-or-degrade decision belongs at this seam.
                    raise
                unreachable.extend(
                    DesignNodeCitation(
                        file_key=file_key,
                        node_id=item.node_id,
                        label=item.label,
                        source_url=item.source_url,
                        reason=_UNREACHABLE_TOKEN_REFUSED,
                    )
                    for item in group
                )
                continue
            files[file_key] = DesignSourceFile(
                file_key=file_key,
                file_name=result.file_name,
                file_version=result.file_version,
            )
            by_id = {item.node_id: item for item in group}
            absent.extend(
                DesignNodeCitation(
                    file_key=file_key,
                    node_id=node_id,
                    label=by_id[node_id].label,
                    source_url=by_id[node_id].source_url,
                    reason=_ABSENT_NOT_IN_FILE,
                )
                for node_id in result.absent_node_ids
            )
            # In the plan's own order, not the provider's: the bound is spent in the order the
            # workstream was told to build, so the frame it was told about first is the frame
            # that fits. Figma answers `nodes` as a mapping and its order is not a promise.
            returned = {subtree.node_id: subtree for subtree in result.subtrees}
            for item in group:
                subtree = returned.get(item.node_id)
                if subtree is None:
                    continue
                content, rendered, depth, characters, built_to = self._deepest_that_fits(subtree)
                reason = self._omission_reason(characters, spent=spent)
                if reason is not None:
                    omitted.append(
                        DesignNodeOmission(
                            file_key=file_key,
                            node_id=item.node_id,
                            label=item.label,
                            reason=reason,
                            characters=characters,
                        )
                    )
                    continue
                spent += characters
                nodes.append(
                    DesignNodeRecord(
                        file_key=file_key,
                        node_id=item.node_id,
                        label=item.label,
                        source_url=item.source_url,
                        applies_to=list(item.applies_to),
                        content=content,
                        characters=characters,
                        nodes_rendered=rendered,
                        depth_rendered=depth,
                        build_depth_rendered=built_to,
                    )
                )
        # Exported after the nodes are rendered and before the artifact is built, because an
        # export stamps `asset_path` into the rendering it came from: the file and the box that
        # needs it are then one fact, rather than two lists a model has to join by name.
        assets, assets_omitted = await self._export_assets(
            client, feature_id=feature_id, nodes=nodes, files=files
        )
        return _detail(
            feature_id=feature_id,
            repository_id=repository_id,
            workstream_id=workstream_id,
            snapshot_artifact_id=snapshot_artifact_id,
            artifact_id=artifact_id,
            files=list(files.values()),
            nodes=nodes,
            assets=assets,
            assets_omitted=assets_omitted,
            omitted=omitted,
            absent=absent,
            unreachable=unreachable,
            characters_selected=spent,
            resolution_mode="live",
            node_character_bound=self._node_character_bound,
            total_character_bound=self._total_character_bound,
            depth_bound=self._depth_bound,
        )

    async def _citations_to_read(
        self, client: FigmaDesignClient, references: Sequence[DesignReference]
    ) -> list[_Citation]:
        """Expand every citation into the concrete nodes to read, whole-file ones included."""
        citations: list[_Citation] = []
        for reference in references:
            if not reference.cites_whole_file:
                citations.extend(
                    _Citation(
                        file_key=reference.file_key,
                        node_id=node_id,
                        label=reference.label,
                        source_url=reference.url,
                        applies_to=tuple(reference.applies_to),
                    )
                    for node_id in reference.node_ids
                )
                continue
            try:
                listed = await client.fetch_top_level_frames(
                    reference.file_key, limit=self._node_bound
                )
            except FigmaClientError as error:
                if not error.is_the_persons_to_fix:
                    raise
                # A whole file nobody can open names no nodes, so it is recorded against the
                # citation itself rather than against frames this platform never learned about.
                citations.append(
                    _Citation(
                        file_key=reference.file_key,
                        node_id=reference.file_key,
                        label=reference.label,
                        source_url=reference.url,
                        applies_to=tuple(reference.applies_to),
                    )
                )
                continue
            citations.extend(
                _Citation(
                    file_key=reference.file_key,
                    node_id=frame.node_id,
                    # The author's label if they gave one, else the frame's own name, which is
                    # more use to a reader than a node id.
                    label=reference.label or frame.name,
                    source_url=reference.url,
                    applies_to=tuple(reference.applies_to),
                )
                for frame in listed.frames
            )
        return citations

    def _deepest_that_fits(
        self, subtree: FigmaNodeSubtree
    ) -> tuple[dict[str, Any], int, int, int, int]:
        """Render one frame as deeply as the character budget allows, never all-or-nothing.

        The bound used to be a cliff: a frame that rendered past `_node_character_bound` at the
        configured depth was omitted entirely, so a *larger* screen reached the Engineer as
        nothing at all while a smaller one reached it whole. That is the worst way to spend a
        budget -- the frames most worth seeing are the ones that vanish -- and it is what made
        raising `MAX_DESIGN_DETAIL_DEPTH` unsafe: measured on AB-Feature-229's login frame,
        depth 10 renders 276,188 characters and depth 12 renders 573,536, so a bound of 450,000
        turns one step deeper into a silently absent design.

        Shallower renderings are tried in order until one fits, so the frame arrives at the
        deepest fidelity its budget can pay for. Only a frame that will not fit even at
        `_MIN_DETAIL_DEPTH` is omitted, and it is omitted carrying the characters of that
        smallest attempt, which is what the reason line quotes.

        Returns the content, the node count, the depth reached, and the characters -- measured
        with `_characters`, the indented rendering a model is actually handed, never a compact
        one. See `design-sizes-are-measured-indented`: a compact measurement understates the
        real cost by roughly 2.6x and would re-introduce the cliff it is here to remove.
        """
        attempt = self._depth_bound
        content, rendered, depth = extract_design_node(
            subtree, depth_bound=self._index_tail_depth, build_depth=attempt
        )
        characters = _characters(content)
        while characters > self._node_character_bound and attempt > _MIN_DETAIL_DEPTH:
            # Halving rather than stepping by one: the tree grows superlinearly with depth, so
            # a step at a time spends several renderings to learn what one tells us. Never
            # below `_MIN_DETAIL_DEPTH`, which is the depth this platform shipped for a year.
            attempt = max(_MIN_DETAIL_DEPTH, attempt - _DETAIL_DEPTH_STEP)
            content, rendered, depth = extract_design_node(
                subtree, depth_bound=self._index_tail_depth, build_depth=attempt
            )
            characters = _characters(content)
        return content, rendered, depth, characters, attempt

    async def _export_assets(
        self,
        client: FigmaDesignClient,
        *,
        feature_id: str,
        nodes: Sequence[DesignNodeRecord],
        files: Mapping[str, DesignSourceFile],
    ) -> tuple[list[DesignAsset], list[DesignAssetOmission]]:
        """Render every image the design fills a node with, and say where each one was put.

        Figma stores an image behind an `imageRef` that no REST caller can resolve to bytes, so
        the only way to obtain one is to render the node that carries it -- which is what the
        console preview already does, through the same two calls, for a human looking at a
        screen. Nothing but the wiring is new here.

        Why it matters: without it an image reaches the Engineer as `{"type": "IMAGE"}` beside a
        width and a height, and an empty box is the only honest thing it can draw. AB-Feature-231
        drew exactly that for a 862x868 background and five ad thumbnails.

        Bounded by `MAX_DESIGN_ASSETS` and every refusal reported, never silent: an unreported
        absence reads as "the design had no image there", and the Engineer is then blamed for a
        box nothing was ever given to fill. A render that fails costs that one asset and not the
        resolution -- a design with no artwork is still a design worth building against.
        """
        if self._asset_sink is None:
            return [], []
        wanted = [
            (record, node) for record in nodes for node in _image_filled_nodes(record.content)
        ]
        assets: list[DesignAsset] = []
        omitted: list[DesignAssetOmission] = []
        # The frame itself, as a picture, for the roles that build and judge it. Rendered
        # before its descendants so that a per-feature attachment cap spent on artwork can
        # never cost the one image that says what the screen looks like.
        for record in nodes:
            known = files.get(record.file_key)
            if known is None or not known.file_version:
                continue
            try:
                url = await client.render_preview(
                    record.file_key, record.node_id, version=known.file_version
                )
                picture = await client.fetch_rendered_bytes(url)
            except FigmaClientError:
                # A frame with no picture is the behaviour every run had until now.
                continue
            stored_preview = await self._asset_sink(
                feature_id, _asset_filename(record.node_id, "preview"), picture
            )
            if stored_preview:
                record.preview_attachment_id = stored_preview
        for record, node in wanted:
            node_id = str(node.get("node_id") or "")
            name = str(node.get("name") or "")
            raw_size = node.get("size")
            size: Mapping[str, Any] = raw_size if isinstance(raw_size, Mapping) else {}
            width, height = int(size.get("width") or 0), int(size.get("height") or 0)
            if len(assets) >= MAX_DESIGN_ASSETS:
                omitted.append(
                    DesignAssetOmission(
                        node_id=node_id or record.node_id,
                        name=name,
                        reason=_ASSET_OVER_COUNT_BOUND,
                        width=width,
                        height=height,
                    )
                )
                continue
            known = files.get(record.file_key)
            version = known.file_version if known is not None else ""
            if not version:
                # Safety rule 4: Figma's images endpoint renders the file's *current* state
                # unless pinned, so an unpinned render would show a design that changed under
                # an unchanged snapshot -- the exact drift the resolve-once rule exists to
                # prevent. Refused and named rather than rendered unpinned.
                omitted.append(
                    DesignAssetOmission(
                        node_id=node_id or record.node_id,
                        name=name,
                        reason=_ASSET_VERSION_UNKNOWN,
                        width=width,
                        height=height,
                    )
                )
                continue
            try:
                url = await client.render_preview(record.file_key, node_id, version=version)
                content = await client.fetch_rendered_bytes(url)
            except FigmaClientError as error:
                # Named with the platform's own classification and never a provider message.
                omitted.append(
                    DesignAssetOmission(
                        node_id=node_id or record.node_id,
                        name=name,
                        reason=error.error_code,
                        width=width,
                        height=height,
                    )
                )
                continue
            stored = await self._asset_sink(feature_id, _asset_filename(node_id, name), content)
            path = f"{_ASSET_DIRECTORY}/{_asset_filename(node_id, name)}"
            assets.append(
                DesignAsset(
                    node_id=node_id or record.node_id,
                    name=name,
                    media_type="image/png",
                    workspace_path=path,
                    attachment_id=stored or "",
                    width=width,
                    height=height,
                    bytes_written=len(content),
                )
            )
            # Stamped into the rendering the Engineer reads, so the file and the box that needs
            # it are one fact rather than two lists a model has to join by name.
            node["asset_path"] = path
        return assets, omitted

    def _omission_reason(self, characters: int, *, spent: int) -> str | None:
        """Say why this node cannot be quoted whole, or ``None`` when it can.

        The bound is spent in citation order and a node that does not fit costs only itself:
        a later, smaller frame is still quoted when it fits in what remains. Dropping
        everything after the first oversized frame would hide small frames for no reason other
        than the order the person happened to paste them in -- `contract_sections`' rule.
        """
        if characters > self._node_character_bound:
            return _OMITTED_OVER_NODE_BOUND
        if spent + characters > self._total_character_bound:
            return _OMITTED_OVER_TOTAL_BOUND
        return None

    def _index_omission_reason(self, characters: int, *, spent: int) -> str | None:
        """Say why this node cannot even be *indexed*, or ``None`` when it can.

        Should be answered ``None`` for every real design ever cited, and that is the design
        rather than an aspiration: the index bounds are runaway guards, sized 2.6x above the
        largest frame measured and 3.5x above a thirty-screen flow. A frame missing from the
        index is a frame the planner cannot assign and therefore a screen the feature silently
        does not build, so these bounds biting is a worse outcome than a large prompt.

        `index_total_exhausted` is its own reason rather than the detail tier's
        `over_total_character_bound`, so a reader can tell the two apart at a glance: one means
        "this flow is enormous", the other means "this workstream already has enough screens".
        """
        if characters > self._index_node_character_bound:
            return _OMITTED_OVER_INDEX_NODE_BOUND
        if spent + characters > self._index_total_character_bound:
            return _OMITTED_INDEX_TOTAL_EXHAUSTED
        return None


class _Citation:
    """One concrete node to read, and the citation it came from."""

    __slots__ = ("applies_to", "file_key", "label", "node_id", "source_url")

    def __init__(
        self,
        *,
        file_key: str,
        node_id: str,
        label: str,
        source_url: str,
        applies_to: tuple[str, ...],
    ) -> None:
        """Carry the node and its provenance together, so neither is looked up twice."""
        self.file_key = file_key
        self.node_id = node_id
        self.label = label
        self.source_url = source_url
        self.applies_to = applies_to


def extract_design_node(
    subtree: FigmaNodeSubtree,
    *,
    depth_bound: int = MAX_DESIGN_DEPTH,
    detail: DesignRenderMode = "build",
    build_depth: int | None = None,
) -> tuple[dict[str, Any], int, int]:
    """Render one node's subtree as text, and report how much of it that was.

    Returns the rendered content, the number of nodes in it, and the depth reached. The wire
    shape stops here: this is the only reader of `FigmaNodeSubtree.document`.

    ``detail`` selects which of the two questions the rendering answers. ``"build"`` is the
    rendering this platform has always produced and the only one an engineer may be given.
    ``"index"`` answers "what frames are there, what are they called, what do they say" -- for
    the product manager deriving requirements and the planner deciding which repository each
    frame belongs to. Measured at ~450 characters a node against ~1,470 at build fidelity.

    The two are different renderings and not a full one and a trimmed one. An index record
    must never reach the Engineer, its self-review or the Reviewer *as a whole frame*: it
    carries no paint, no typography and no layout, and a model asked to build from that
    invents values that pass every gate this platform has, because no command it runs checks a
    colour.

    ``build_depth`` is the hybrid, and the one exception to that rule -- a *deeper* index tail
    under a build root, never an index frame standing alone. `design_content_is_buildable`
    still passes, because the root and everything down to `build_depth` carries paint and
    geometry; what the tail adds is the words. See `_render` for the measurement that made it
    necessary: a flat depth-10 build rendering gave the Engineer 24 of 71 text strings, and the
    screen it produced was a set of correctly-positioned empty boxes.
    """
    content = _render(
        subtree.document,
        subtree=subtree,
        path="",
        depth=0,
        depth_bound=depth_bound,
        detail=detail,
        build_depth=build_depth,
    )
    rendered, deepest = _walk(content)
    return content, rendered, deepest


def _render(
    node: Mapping[str, Any],
    *,
    subtree: FigmaNodeSubtree,
    path: str,
    depth: int,
    depth_bound: int,
    detail: DesignRenderMode = "build",
    build_depth: int | None = None,
) -> dict[str, Any]:
    """Render one node, names first, then its children in document order.

    Names before values in both modes. In index mode there are no values, which is the point:
    what survives is exactly what somebody deciding *where a frame belongs* reads, and nothing
    somebody implementing it would need.

    ``build_depth`` makes a *hybrid* rendering: build fidelity down to that depth and index
    fidelity below it, to the full `depth_bound`. It exists because of what a flat bound
    actually costs. Measured on AB-Feature-231's login frame, a build rendering bounded at
    depth 10 carried 24 of the design's 71 text strings -- the other 47 live at the leaves, and
    the implementation it produced drew every container as an empty box because the words
    inside them had been cut off. Paying build fidelity for those leaves is unaffordable
    (754,605 characters at depth 14, about 188,000 tokens against a 272,000 window), but an
    index record costs roughly 450 characters against 1,470, so the *words* are affordable even
    where the geometry is not.

    ``None`` keeps the flat behaviour, so the index tier and every existing caller render
    byte-for-byte what they rendered before.
    """
    building = detail == "build" and (build_depth is None or depth <= build_depth)
    # Below `build_depth` the rendering carries the *words* and nothing else. The tail exists
    # because a flat build bound cut 47 of the login frame's 71 text strings and the screen
    # came back as empty boxes -- it does not exist to carry a second copy of the structure.
    # Measured: a full index tail costs 186,802 characters on that frame, and the text-only
    # tail costs a fraction of it, which is the difference between the design fitting beside
    # the repository snapshot in a 272,000-token window and not fitting.
    tail = detail == "build" and build_depth is not None and depth > build_depth
    name = str(node.get("name") or "")
    here = f"{path}/{name}" if path else name
    out: dict[str, Any] = (
        {"name": name, "type": node.get("type")}
        if tail
        else {"path": here, "name": name, "type": node.get("type")}
    )
    # The node's own Figma id. Carried in both tiers because two different readers need it and
    # neither can reconstruct it: a person goes back to the exact rectangle in the file with it,
    # and the asset export renders *by node id* -- without it every image fill was requested as
    # `render_preview(key, "")` and Figma answered 400 six times, which is how AB-Feature-232
    # exported nothing.
    node_id = node.get("id")
    if isinstance(node_id, str) and node_id and not tail:
        out["node_id"] = node_id
    box = node.get("absoluteBoundingBox")
    if building and isinstance(box, dict):
        out["size"] = {
            "width": _rounded(box.get("width")),
            "height": _rounded(box.get("height")),
        }
    # Names before values, deliberately and in this order: a style name says what the
    # repository already has, and a hex code says what to hardcode.
    style_names = {} if tail else _style_names(node, subtree)
    if style_names:
        out["style_names"] = style_names
    if not tail and node.get("type") in _COMPONENT_NODE_TYPES:
        component = _component(node, subtree)
        if component:
            out["component"] = component
        variants = _variant_properties(node)
        if variants:
            out["variant_properties"] = variants
    if building:
        layout = {key: node.get(key) for key in _LAYOUT_KEYS if node.get(key) is not None}
        if layout:
            out["layout"] = layout
        constraints = node.get("constraints")
        if isinstance(constraints, dict):
            out["constraints"] = constraints
    if node.get("type") == "TEXT":
        # The actual characters, never a summary of them: "the text is the text" is one of the
        # few design criteria a reviewer reading a diff can actually check. Kept in the index
        # too, and deliberately: the copy on a screen is often the only thing that says which
        # screen it is, which is exactly what the index is asked.
        out["text"] = node.get("characters")
        if building:
            typography = _typography(node)
            if typography:
                out["typography"] = typography
    if building:
        fills = _paints(node.get("fills"))
        if fills:
            out["fills"] = fills
        strokes = _paints(node.get("strokes"))
        if strokes:
            out["strokes"] = strokes
            if node.get("strokeWeight") is not None:
                out["stroke_weight"] = node.get("strokeWeight")
        effects = _effects(node.get("effects"))
        if effects:
            out["effects"] = effects
        if node.get("cornerRadius") is not None:
            out["corner_radius"] = node.get("cornerRadius")
        if node.get("rectangleCornerRadii") is not None:
            out["corner_radii"] = node.get("rectangleCornerRadii")
    children = [item for item in (node.get("children") or []) if isinstance(item, dict)]
    if not children:
        return out
    if depth >= depth_bound:
        # A reported cap. A depth cut that silently dropped children would read as a frame
        # that has none, which is exactly the misreading the omission lists exist to prevent.
        out["children_beyond_depth_bound"] = len(children)
        return out
    out["children"] = [
        _render(
            child,
            subtree=subtree,
            path=here,
            depth=depth + 1,
            depth_bound=depth_bound,
            detail=detail,
            build_depth=build_depth,
        )
        for child in children
    ]
    return out


def _image_filled_nodes(content: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Yield every rendered node the design fills with an image, root included.

    Read off the *rendering* rather than the raw Figma tree, so a node cut off by the depth
    bound is never exported: the Engineer cannot reference a box it was not shown.
    """
    found: list[dict[str, Any]] = []

    def walk(node: Any) -> None:
        if not isinstance(node, dict):
            return
        fills = node.get("fills")
        painted = fills if isinstance(fills, list) else []
        if any(isinstance(f, Mapping) and f.get("type") == "IMAGE" for f in painted):
            found.append(node)
        children = node.get("children")
        for child in children if isinstance(children, list) else []:
            walk(child)

    walk(content)
    return found


def _asset_filename(node_id: str, name: str) -> str:
    """A filename that is stable across resumes and safe on every filesystem.

    Keyed on the node id rather than the author's layer name, because two layers may share a
    name and a name may contain anything at all. The name rides along only so a person reading
    a diff can tell which rectangle it was.
    """
    safe_id = "".join(ch if ch.isalnum() else "-" for ch in node_id).strip("-") or "node"
    safe_name = "".join(ch if ch.isalnum() else "-" for ch in name.lower()).strip("-")
    return f"{safe_id}-{safe_name}.png"[:120] if safe_name else f"{safe_id}.png"


def _style_names(node: Mapping[str, Any], subtree: FigmaNodeSubtree) -> dict[str, str]:
    """Map each style slot this node references to the style's own name."""
    references = node.get("styles")
    if not isinstance(references, dict):
        return {}
    named: dict[str, str] = {}
    for slot, style_id in references.items():
        entry = subtree.styles.get(str(style_id))
        if isinstance(entry, dict) and entry.get("name"):
            named[str(slot)] = str(entry["name"])
    return named


def _component(node: Mapping[str, Any], subtree: FigmaNodeSubtree) -> dict[str, Any]:
    """Name the component this node is, or is an instance of, and its set.

    A variant component's own `name` is its variant string ("Size=M, State=Default"), so the
    set's name is what says *which component* it is -- and reusing the component the design
    names rather than re-implementing it is one of the criteria a diff can be judged against.
    """
    component_id = str(node.get("componentId") or node.get("id") or "")
    entry = subtree.components.get(component_id)
    if not isinstance(entry, dict):
        return {}
    named: dict[str, Any] = {"name": entry.get("name")}
    set_entry = subtree.component_sets.get(str(entry.get("componentSetId") or ""))
    if isinstance(set_entry, dict) and set_entry.get("name"):
        named["set"] = set_entry["name"]
    return named


def _variant_properties(node: Mapping[str, Any]) -> dict[str, Any]:
    """Read the variant properties an instance sets, which are the states it is in."""
    properties = node.get("componentProperties")
    if not isinstance(properties, dict):
        return {}
    return {
        str(key): value.get("value") for key, value in properties.items() if isinstance(value, dict)
    }


def _typography(node: Mapping[str, Any]) -> dict[str, Any]:
    """Read the type this text is set in, as declared."""
    style = node.get("style")
    if not isinstance(style, dict):
        return {}
    return {key: style[source] for key, source in _TYPOGRAPHY_KEYS if style.get(source) is not None}


def _paints(value: Any) -> list[dict[str, Any]]:
    """Reduce a paint list to what a person writing a component would read."""
    if not isinstance(value, list):
        return []
    reduced: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("type") or "")
        if item.get("visible") is False:
            continue
        entry: dict[str, Any] = {"type": kind}
        colour = item.get("color")
        if kind == "SOLID" and isinstance(colour, dict):
            entry["hex"] = _hex(colour)
            alpha = colour.get("a")
            if isinstance(alpha, int | float) and float(alpha) != 1.0:
                entry["alpha"] = round(float(alpha), 3)
        opacity = item.get("opacity")
        if isinstance(opacity, int | float) and float(opacity) != 1.0:
            entry["opacity"] = round(float(opacity), 3)
        reduced.append(entry)
    return reduced


def _effects(value: Any) -> list[dict[str, Any]]:
    """Read the effects as declared -- a shadow's kind and radius, not its rendering."""
    if not isinstance(value, list):
        return []
    return [
        {
            key: item[source]
            for key, source in (("type", "type"), ("radius", "radius"), ("spread", "spread"))
            if item.get(source) is not None
        }
        for item in value
        if isinstance(item, dict) and item.get("visible") is not False
    ]


def _hex(colour: dict[str, Any]) -> str | None:
    """Render one Figma colour as the hex an engineer would otherwise have to compute."""
    try:
        channels = [round(255 * float(colour.get(name, 0))) for name in ("r", "g", "b")]
    except (TypeError, ValueError):
        return None
    return "#" + "".join(f"{max(0, min(255, channel)):02x}" for channel in channels)


def _rounded(value: Any) -> float | None:
    """Round a measurement to a tenth, which is the precision a spacing scale is written in."""
    if not isinstance(value, int | float):
        return None
    return round(float(value), 1)


def _walk(content: dict[str, Any]) -> tuple[int, int]:
    """Count the nodes in a rendered tree and the depth it reaches."""
    count = 1
    deepest = 0
    for child in content.get("children") or []:
        if not isinstance(child, dict):
            continue
        child_count, child_depth = _walk(child)
        count += child_count
        deepest = max(deepest, child_depth + 1)
    return count, deepest


def _characters(content: dict[str, Any]) -> int:
    """Measure one rendered node the way it will actually be handed to a model.

    The same serialization the selection uses, so the number the bound is compared against is
    the number of characters that would reach a prompt -- not an approximation of it.
    """
    import json

    return len(json.dumps(content, indent=2, sort_keys=True, ensure_ascii=False))


def _by_file(citations: Sequence[_Citation]) -> dict[str, list[_Citation]]:
    """Group the nodes to read by file, so each file costs one call rather than one per node."""
    grouped: dict[str, list[_Citation]] = {}
    for citation in citations:
        grouped.setdefault(citation.file_key, []).append(citation)
    return grouped


def _provenance_by_node(references: Sequence[DesignReference]) -> dict[str, _Citation]:
    """Map each frame a workstream can be assigned to the citation it came from.

    Detail resolution is handed frame ids by the plan, not citations, so this is how a frame
    id becomes a file key, a label, the URL the person pasted and the repositories they scoped
    it to. Built from the citations rather than from the snapshot's records because a citation
    is what the person actually wrote, and the label they gave is the one a prompt should show.

    A whole-file citation names no frames, so its frames -- listed from the file at resolution
    time -- cannot be found here by id. They are attributed to that citation's file anyway,
    which is sound because a whole-file citation carries exactly one file key: any frame the
    plan names that no explicit citation claims must have come from a file somebody cited
    whole. With more than one such citation the first is used and the label is the author's,
    which is the same answer `_citations_to_read` reaches by a different route.
    """
    explicit: dict[str, _Citation] = {}
    whole_file: list[DesignReference] = []
    for reference in references:
        if reference.cites_whole_file:
            whole_file.append(reference)
            continue
        for node_id in reference.node_ids:
            explicit.setdefault(
                node_id,
                _Citation(
                    file_key=reference.file_key,
                    node_id=node_id,
                    label=reference.label,
                    source_url=reference.url,
                    applies_to=tuple(reference.applies_to),
                ),
            )
    if not whole_file:
        return explicit
    return _WholeFileProvenance(explicit, whole_file[0])


class _WholeFileProvenance(dict[str, _Citation]):
    """Explicit citations, falling back to the one file somebody cited whole.

    A mapping rather than a function so `resolve_detail` reads the same either way. Subclassing
    `dict` keeps `in` honest for the explicit frames while `get` answers for the listed ones --
    which is what a whole-file citation needs, since its frames were never written down.
    """

    def __init__(self, explicit: dict[str, _Citation], fallback: DesignReference) -> None:
        """Bind the explicit map and the citation every other frame is attributed to."""
        super().__init__(explicit)
        self._fallback = fallback

    def __contains__(self, key: object) -> bool:
        """Every frame is accounted for: explicitly, or by the file cited whole."""
        return isinstance(key, str)

    def get(self, key: str, default: Any = None) -> Any:
        """Return the explicit citation, or one derived from the whole-file citation."""
        found = super().get(key)
        if found is not None:
            return found
        return _Citation(
            file_key=self._fallback.file_key,
            node_id=key,
            label=self._fallback.label,
            source_url=self._fallback.url,
            applies_to=tuple(self._fallback.applies_to),
        )

    def __getitem__(self, key: str) -> _Citation:
        """Index the same way `get` answers, so a caller cannot see the two disagree."""
        return cast("_Citation", self.get(key))


def _unreachable_citation(node_id: str, provenance: dict[str, _Citation]) -> DesignNodeCitation:
    """Record one assigned frame as unreachable, keeping whatever provenance is known."""
    citation = provenance.get(node_id)
    return DesignNodeCitation(
        file_key=citation.file_key if citation is not None else node_id,
        node_id=node_id,
        label=citation.label if citation is not None else "",
        source_url=citation.source_url if citation is not None else "",
        reason=_UNREACHABLE_TOKEN_REFUSED,
    )


def _no_nodes_diagnostic(
    absent: Sequence[DesignNodeCitation], unreachable: Sequence[DesignNodeCitation]
) -> str:
    """Say which of the two answers happened, in words a person can act on."""
    if unreachable:
        keys = ", ".join(dict.fromkeys(item.file_key for item in unreachable))
        return (
            f"This deployment's Figma token could not read {keys}. Either the link points at a "
            "file the account cannot open, or the token needs replacing in Settings."
        )
    if absent:
        frames = ", ".join(f"{item.file_key}#{item.node_id}" for item in absent)
        return (
            f"Figma does not define the cited frame(s) {frames}. The frame may have been "
            "deleted or renamed away; copy the link again from the frame you meant."
        )
    return (
        "No design could be resolved from this feature's citations. Re-copy the links from "
        "Figma and resume."
    )


def _snapshot(
    *,
    feature_id: str,
    artifact_id: str,
    files: list[DesignSourceFile],
    nodes: list[DesignNodeRecord],
    omitted: list[DesignNodeOmission],
    absent: list[DesignNodeCitation],
    unreachable: list[DesignNodeCitation],
    characters_selected: int,
    resolution_mode: str,
    node_bound: int = MAX_DESIGN_NODES,
    node_character_bound: int = MAX_DESIGN_DETAIL_NODE_CHARACTERS,
    total_character_bound: int = MAX_DESIGN_DETAIL_TOTAL_CHARACTERS,
    depth_bound: int = MAX_DESIGN_DETAIL_DEPTH,
) -> DesignSnapshotArtifact:
    """Build the one artifact both resolvers produce, so their contract cannot drift."""
    return create_artifact(
        DesignSnapshotArtifact,
        workflow_id=feature_id,
        artifact_id=artifact_id,
        producer="design_resolver",
        payload={
            "feature_id": feature_id,
            "resolved_at": datetime.now(UTC),
            "files": [item.model_dump(mode="python") for item in files],
            "nodes": [item.model_dump(mode="python") for item in nodes],
            "design_nodes_omitted": [item.model_dump(mode="python") for item in omitted],
            "design_nodes_absent": [item.model_dump(mode="python") for item in absent],
            "design_nodes_unreachable": [item.model_dump(mode="python") for item in unreachable],
            "style_name_source": STYLE_NAMES_FROM_NODES,
            "bounds": {
                "nodes": node_bound,
                "characters_per_node": node_character_bound,
                "characters_total": total_character_bound,
                "depth": depth_bound,
            },
            "characters_selected": characters_selected,
        },
        metadata={"resolution_mode": resolution_mode},
    )


def _detail(
    *,
    feature_id: str,
    repository_id: str,
    workstream_id: str,
    snapshot_artifact_id: str,
    artifact_id: str,
    files: list[DesignSourceFile],
    nodes: list[DesignNodeRecord],
    omitted: list[DesignNodeOmission],
    absent: list[DesignNodeCitation],
    unreachable: list[DesignNodeCitation],
    characters_selected: int,
    resolution_mode: str,
    node_character_bound: int = MAX_DESIGN_DETAIL_NODE_CHARACTERS,
    total_character_bound: int = MAX_DESIGN_DETAIL_TOTAL_CHARACTERS,
    depth_bound: int = MAX_DESIGN_DETAIL_DEPTH,
    assets: Sequence[DesignAsset] = (),
    assets_omitted: Sequence[DesignAssetOmission] = (),
) -> DesignDetailArtifact:
    """Build the one detail artifact both resolvers produce, so their contract cannot drift.

    Beside `_snapshot` and for its reason: two resolvers writing the same artifact by hand is
    two chances for the mock tier to be testing a shape the live path does not produce.
    """
    return create_artifact(
        DesignDetailArtifact,
        workflow_id=feature_id,
        artifact_id=artifact_id,
        producer="design_resolver",
        payload={
            "feature_id": feature_id,
            "repository_id": repository_id,
            "workstream_id": workstream_id,
            "snapshot_artifact_id": snapshot_artifact_id,
            "resolved_at": datetime.now(UTC),
            "files": [item.model_dump(mode="python") for item in files],
            "nodes": [item.model_dump(mode="python") for item in nodes],
            "design_nodes_omitted": [item.model_dump(mode="python") for item in omitted],
            "design_nodes_absent": [item.model_dump(mode="python") for item in absent],
            "design_nodes_unreachable": [item.model_dump(mode="python") for item in unreachable],
            "bounds": {
                "characters_per_node": node_character_bound,
                "characters_total": total_character_bound,
                "depth": depth_bound,
            },
            "characters_selected": characters_selected,
            "assets": [item.model_dump(mode="python") for item in assets],
            "assets_omitted": [item.model_dump(mode="python") for item in assets_omitted],
        },
        metadata={"resolution_mode": resolution_mode},
    )


def unreachable_design_detail(
    *,
    feature_id: str,
    repository_id: str,
    workstream_id: str,
    snapshot_artifact_id: str,
    snapshot: DesignSnapshotArtifact,
    node_ids: Sequence[str],
    artifact_id: str,
) -> DesignDetailArtifact:
    """Build the detail artifact for a workstream whose design could not be read at all.

    Every assigned frame recorded unreachable, with whatever the snapshot already knows about
    it -- which file it is in and what the author called it. The provenance comes from the
    snapshot rather than the citations because the snapshot is the authority on what a frame
    id *is*: the planner may only assign frames the snapshot mentions, and a whole-file
    citation's frames were never written down anywhere else.

    This exists so that "the design source was down" is a thing the prompts can say, rather
    than a thing that ends a repository's workstream. See `_resolve_design_detail`.
    """
    known: dict[str, tuple[str, str]] = {}
    for record in snapshot.nodes:
        known.setdefault(record.node_id, (record.file_key, record.label))
    for group in (
        snapshot.design_nodes_omitted,
        snapshot.design_nodes_absent,
        snapshot.design_nodes_unreachable,
    ):
        for item in group:
            known.setdefault(item.node_id, (item.file_key, item.label))
    return _detail(
        feature_id=feature_id,
        repository_id=repository_id,
        workstream_id=workstream_id,
        snapshot_artifact_id=snapshot_artifact_id,
        artifact_id=artifact_id,
        files=[],
        nodes=[],
        omitted=[],
        absent=[],
        unreachable=[
            DesignNodeCitation(
                file_key=known.get(node_id, (node_id, ""))[0],
                node_id=node_id,
                label=known.get(node_id, (node_id, ""))[1],
                reason=_UNREACHABLE_TOKEN_REFUSED,
            )
            for node_id in dict.fromkeys(node_ids)
        ],
        characters_selected=0,
        resolution_mode="unreachable",
    )


def design_snapshot_artifact_id(existing: int) -> str:
    """Return the artifact id for the next resolution, revision-qualified after the first.

    Safety rule 4: nothing re-resolves in place. A refresh is a new revision and the console
    shows both, exactly as it shows PRD revisions -- so the first snapshot a feature was built
    against stays readable forever, which is what makes an unexplainable diff explainable.
    """
    base = FEATURE_ARTIFACT_FILENAMES["design_snapshot"]
    if existing <= 0:
        return base
    return f"{base.removesuffix('.json')}.revision-{existing + 1}.json"


def design_detail_artifact_id(repository_id: str, existing: int) -> str:
    """Return the artifact id for one repository's detail, revision-qualified after the first.

    Keyed by repository in the shape `design_snapshot_artifact_id` establishes, and keyed by it
    for a reason the snapshot does not have: several of these exist per feature at once, one per
    workstream, and two of them landing on one filename would mean one workstream's frames
    silently replacing another's -- which the reader would see as a workstream that was handed
    the wrong screens.

    Revision-qualified on the same rule as the snapshot (Safety rule 4): a re-resolution is a
    new revision and never an edit, so the detail an attempt was actually given stays readable
    after a refresh.
    """
    base = FEATURE_ARTIFACT_FILENAMES["design_detail"].removesuffix(".json")
    stem = f"{base}.{repository_id}"
    if existing <= 0:
        return f"{stem}.json"
    return f"{stem}.revision-{existing + 1}.json"


__all__ = [
    "DESIGN_BUILD_ONLY_KEYS",
    "DESIGN_INDEX_KEYS",
    "STYLE_NAMES_FROM_NODES",
    "DesignReferenceResolver",
    "DesignRenderMode",
    "DesignResolutionRefused",
    "DeterministicDesignResolver",
    "FigmaDesignResolver",
    "design_content_is_buildable",
    "design_detail_artifact_id",
    "design_snapshot_artifact_id",
    "extract_design_node",
    "unreachable_design_detail",
]
