"""The resolved design's own text, selected for every role that builds or judges the work.

Five roles are asked to build or rule on a screen -- the product manager, the planner, the
Engineer, its self-review pass, and the independent Reviewer -- and until this module existed
all five were asked while holding prose alone. A reviewer holding the design can say "this uses
a raw hex where the design names a token"; a reviewer holding nothing has been saying "looks
reasonable" for the entire life of this platform.

Built in `contract_sections.py`'s shape and spirit, and for its reason: three judges sharing one
blindness must not become three judges with three slightly different sights. One function, one
JSON rendering, read from state, given to everybody -- so the Engineer, its self-review and the
Reviewer cannot be handed different selections of the same design for the same attempt.

Two properties, and one deliberate difference from the precedent.

**Absent is a fact, not a gap.** A workstream whose plan names a frame the resolution omitted is
told so, with the reason and the size, rather than being handed a design that quietly has one
fewer screen in it than the plan says. Every one of the snapshot's three lists is carried
through, filtered to this workstream's own frames.

**Names outrank values.** The snapshot already renders each frame with its style, component and
component-set names ahead of its resolved fills; this module changes nothing about that and
adds no summary of its own. A judge reading it is reading the design.

**The bound is not re-applied here.** `contract_section_context` bounds at selection time
because the contract it quotes is unbounded. A design snapshot is already bounded -- at
resolution, once, in an immutable artifact that reports exactly what it dropped -- so bounding
again would create a second place a frame can silently disappear, with two reports to reconcile.
What this module does instead is *carry* the snapshot's bounds and its omissions, so the reader
sees one account of what was left out and one set of numbers behind it.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any

from artifacts.design_references import design_content_is_buildable
from artifacts.schemas import (
    DesignDetailArtifact,
    DesignNodeRecord,
    DesignSnapshotArtifact,
    TaskPlanArtifact,
)
from state.models import AgentState

# Where a workstream's assignment is read from. The review scope, not the workstream plan,
# for `scoped_section_relationships`' reason: the review scope is what both judges are bounded
# by, and reading the bound from anywhere else would let the quoted design and the assignment
# disagree.
_SCOPE_FIELD = "design_nodes"

# Why a frame the flow definitely has is not quoted to this workstream. Distinct from every
# bound reason: no bound bit, there was simply no build-fidelity resolution to quote. Its own
# word so a reader is never left thinking a bound hid a screen that nothing ever read.
_DETAIL_NOT_RESOLVED = "detail_not_resolved_for_this_workstream"


def design_node_names(snapshot: DesignSnapshotArtifact) -> frozenset[str]:
    """Every frame this snapshot mentions, however it mentions it.

    The authority a planner assignment is checked against, and it deliberately includes the
    frames that were omitted, absent or unreachable as well as the ones quoted. A frame the
    resolution could not fit is still a frame the design has and a workstream may legitimately
    be assigned -- it is told about the omission. A frame the snapshot has never heard of is
    one the planner invented, and that is what the check refuses.

    Shared with the selection below, so "does this design contain that frame" has exactly one
    answer in this codebase: a name the planner accepts is a name this selection can account
    for, and a name it rejects is one that would have reached an engineer as an instruction to
    build a screen nobody drew.
    """
    return frozenset(
        item.node_id
        for group in (
            snapshot.nodes,
            snapshot.design_nodes_omitted,
            snapshot.design_nodes_absent,
            snapshot.design_nodes_unreachable,
        )
        for item in group
    )


def design_snapshot_in_state(state: AgentState) -> DesignSnapshotArtifact | None:
    """Return the latest same-workflow design snapshot in this state, or nothing.

    `None` for every feature that cited no design, which is most of them -- and callers render
    nothing for it, so those runs' prompts are unchanged byte for byte.
    """
    for artifact in reversed(state["artifacts"]):
        if (
            isinstance(artifact, DesignSnapshotArtifact)
            and artifact.workflow_id == state["workflow_id"]
        ):
            return artifact
    return None


def scoped_design_node_ids(task_plan: TaskPlanArtifact) -> tuple[str, ...]:
    """Return the frames this task plan's review scope assigns, in the plan's own order."""
    scope = task_plan.metadata.get("review_scope")
    if not isinstance(scope, Mapping):
        return ()
    return tuple(_clean_names(scope.get(_SCOPE_FIELD)))


def design_request_context(snapshot: DesignSnapshotArtifact | None) -> dict[str, Any] | None:
    """Select the whole design, for the two roles that read it as part of the request.

    The product manager and the planner both see everything, because neither has a workstream
    yet: the product manager is deriving requirements from the frames, and the planner is the
    one deciding which repository each frame belongs to. Scoping either of them would be
    scoping by a decision that has not been made.

    **And they see the deciding rendering, not the building one.** What the snapshot carries is
    the index: structure, names and text, with no layout, typography or paint. That is not a
    reduction for these two roles, it is the right rendering -- both are choosing which frame
    means what and which repository it belongs to, and neither writes a line of CSS. The
    frames' style and component *names* are exactly what survives, which is what a planner
    needs to say "this screen is the design system's, that one is bespoke".

    It is also what makes a whole flow readable at all: a thirty-screen citation indexes in
    about a third of what it would cost to quote at build fidelity, and the roles that have to
    see all thirty are these two.

    ``None`` rather than an empty block when nothing was cited, so a citation-free feature's
    prompts are byte-identical to what they were before this item.
    """
    if snapshot is None or not snapshot.nodes:
        return None
    return {
        "files": [item.model_dump(mode="json") for item in snapshot.files],
        "resolved_at": snapshot.resolved_at.isoformat(),
        "style_name_source": snapshot.style_name_source,
        "design_nodes": [_quoted(item) for item in snapshot.nodes],
        "design_nodes_omitted": [
            item.model_dump(mode="json") for item in snapshot.design_nodes_omitted
        ],
        "design_nodes_absent": [
            item.model_dump(mode="json") for item in snapshot.design_nodes_absent
        ],
        "design_nodes_unreachable": [
            item.model_dump(mode="json") for item in snapshot.design_nodes_unreachable
        ],
        "bounds": dict(snapshot.bounds),
        "characters_selected": snapshot.characters_selected,
    }


def design_snapshot_context(
    *,
    snapshot: DesignSnapshotArtifact | None,
    detail: DesignDetailArtifact | None,
    task_plan: TaskPlanArtifact,
) -> dict[str, Any] | None:
    """Select the design for one workstream's own frames, or ``None`` if there is none.

    ``None`` for the two cases that are not a selection at all: a feature with no snapshot in
    its state, which is every feature that cited nothing, and a workstream whose review scope
    names no frames -- which is every workstream of a feature whose design applies elsewhere.
    Callers render nothing for it, so those prompts are unchanged byte for byte.

    **The detail artifact is the authority for a quoted frame; the snapshot is only a
    fallback for what detail could not say.** In order, per frame:

    1. in ``detail.nodes`` -- quote the build-fidelity record;
    2. else in one of ``detail``'s omission lists -- carry that omission;
    3. else in one of the snapshot's -- carry that;
    4. else -- ``design_nodes_not_in_snapshot``.

    **Falling back to the snapshot's own record at step 2 or 3 would be a defect, not a
    kindness.** The snapshot holds *index* records: names, structure and text, with no colour,
    no type and no spacing. An engineer handed one of those as a frame to build reads a design
    with the values missing and supplies its own -- and invented values pass every gate this
    platform has, because no command it runs checks a colour. So the honest answer for a frame
    detail could not resolve is the omission, which the prompts already know how to read. That
    is what `_assert_buildable` enforces, by test rather than by care.

    A frame in scope that was omitted, could not be found, or could not be read appears in the
    matching list rather than being silently missing -- the rule the whole item runs on: a
    judge shown nothing knows it was shown nothing.
    """
    if snapshot is None:
        return None
    scoped = scoped_design_node_ids(task_plan)
    if not scoped:
        return None
    # Detail first in every lookup, because detail is what this workstream was actually given.
    quoted_by_id = {item.node_id: item for item in (detail.nodes if detail else ())}
    # The index, used only to tell "the design has no such frame" from "the design has it and
    # this workstream was not given it". Never quoted from -- see `_assert_buildable`.
    indexed_by_id = {item.node_id: item for item in snapshot.nodes}
    omitted_by_id = {
        item.node_id: item
        for group in (
            snapshot.design_nodes_omitted,
            detail.design_nodes_omitted if detail else (),
        )
        for item in group
    }
    absent_by_id = {
        item.node_id: item
        for group in (snapshot.design_nodes_absent, detail.design_nodes_absent if detail else ())
        for item in group
    }
    unreachable_by_id = {
        item.node_id: item
        for group in (
            snapshot.design_nodes_unreachable,
            detail.design_nodes_unreachable if detail else (),
        )
        for item in group
    }
    # A frame the snapshot omitted but detail resolved is a frame this workstream *was* shown:
    # the index bound and the detail bound are different bounds, and the detail one is what
    # governs what an engineer got. So a resolved frame is never also reported missing.
    for node_id in quoted_by_id:
        omitted_by_id.pop(node_id, None)
        absent_by_id.pop(node_id, None)
        unreachable_by_id.pop(node_id, None)
    known_files = {
        item.file_key: item for item in (*snapshot.files, *(detail.files if detail else ()))
    }
    quoted: list[dict[str, Any]] = []
    omitted: list[dict[str, Any]] = []
    absent: list[dict[str, Any]] = []
    unreachable: list[dict[str, Any]] = []
    unknown: list[str] = []
    files: dict[str, dict[str, Any]] = {}
    spent = 0
    for node_id in scoped:
        if node_id in quoted_by_id:
            record = quoted_by_id[node_id]
            _assert_buildable(record)
            quoted.append(_quoted(record))
            spent += record.characters
            source = known_files.get(record.file_key)
            if source is not None:
                files[record.file_key] = source.model_dump(mode="json")
            continue
        if node_id in omitted_by_id:
            omitted.append(omitted_by_id[node_id].model_dump(mode="json"))
            continue
        if node_id in absent_by_id:
            absent.append(absent_by_id[node_id].model_dump(mode="json"))
            continue
        if node_id in unreachable_by_id:
            unreachable.append(unreachable_by_id[node_id].model_dump(mode="json"))
            continue
        if node_id in indexed_by_id:
            # The flow has this frame -- the index lists it -- but no detail resolution ever
            # said anything about it, so this workstream was not given it. Reported as an
            # omission rather than as `not_in_snapshot`, which would be false: the design does
            # contain this screen. And *never* satisfied from the index record sitting right
            # here, which is the tempting wrong answer: that record has no colour, no type and
            # no spacing in it, and an engineer handed it would invent all three.
            #
            # Reached by a state that has no detail artifact at all -- a mock composition, a
            # feature whose state predates the two tiers, a resume that lost it -- rather than
            # by any bound.
            record = indexed_by_id[node_id]
            omitted.append(
                {
                    "file_key": record.file_key,
                    "node_id": record.node_id,
                    "label": record.label,
                    "reason": _DETAIL_NOT_RESOLVED,
                    "characters": 0,
                }
            )
            continue
        # The planner's own validation refuses this before a plan is written, so reaching it
        # means an older plan or a snapshot revision that no longer contains the frame. Named
        # rather than dropped: a workstream told to build a frame nothing knows about should
        # say so in `summary` instead of inventing one.
        unknown.append(node_id)
    return {
        "files": list(files.values()),
        "resolved_at": (detail.resolved_at if detail else snapshot.resolved_at).isoformat(),
        "style_name_source": snapshot.style_name_source,
        "design_nodes": quoted,
        "design_nodes_omitted": omitted,
        "design_nodes_absent": absent,
        "design_nodes_unreachable": unreachable,
        "design_nodes_not_in_snapshot": unknown,
        # The bounds a reader should reconcile the omissions against are the ones that produced
        # them: detail's, where a detail resolution happened. The snapshot's index bounds would
        # explain nothing about a frame the detail tier could not fit.
        "bounds": dict(detail.bounds if detail else snapshot.bounds),
        "characters_selected": spent,
    }


def _assert_buildable(record: DesignNodeRecord) -> None:
    """Refuse to quote a frame to a builder without the values needed to build it.

    Risk 1 of 96-, as an assertion rather than a convention. The snapshot's records are index
    records -- no paint, no typography, no layout -- and a selection bug that let one reach the
    Engineer would not fail any test, any lint, any type check or any validation command,
    because nothing this platform runs checks a colour. It would surface as a screen that looks
    wrong to a person, weeks later, with a green pipeline behind it.

    An `AssertionError` is the right shape: this is unreachable by construction -- only
    `detail.nodes` is ever quoted, and detail is resolved at build fidelity -- so reaching it
    means the selection above was edited into a state where it can no longer be trusted, and
    failing the attempt loudly beats shipping invented styling quietly.
    """
    if not design_content_is_buildable(record.content):
        msg = (
            f"refusing to quote {record.file_key}#{record.node_id} to a builder: the record "
            "carries no layout, typography or paint, so it is an index record. An engineer "
            "given this invents the values, and nothing downstream checks them"
        )
        raise AssertionError(msg)


def design_detail_in_state(state: AgentState) -> DesignDetailArtifact | None:
    """Return the newest design detail this state carries that pairs with its snapshot.

    Paired on ``snapshot_artifact_id`` rather than merely on being present: a snapshot refresh
    makes a new revision, and a detail resolved against the previous one describes a design
    this attempt is no longer judged against. Quoting it would be quoting the wrong design
    while claiming it is the right one.

    No repository filter, and none is needed: a child state carries exactly the one detail
    artifact its own executor put there, selected by repository before it was re-homed. The
    filtering happened where the repository is known, which is the only place it can be.
    """
    snapshot = design_snapshot_in_state(state)
    if snapshot is None:
        return None
    for artifact in reversed(state["artifacts"]):
        if (
            isinstance(artifact, DesignDetailArtifact)
            and artifact.workflow_id == state["workflow_id"]
            and artifact.snapshot_artifact_id == snapshot.artifact_id
        ):
            return artifact
    return None


def design_snapshot_context_from_state(
    state: AgentState, task_plan: TaskPlanArtifact
) -> dict[str, Any] | None:
    """Select from whichever design this workflow's state carries, if any.

    The single entry point the Engineer and the Reviewer both call, on the same two inputs, so
    the two cannot be given different selections of the same design for the same attempt. The
    Engineer's self-review pass is handed the Engineer's rendered instructions verbatim, so it
    is the same selection by construction rather than by a third call.
    """
    return design_snapshot_context(
        snapshot=design_snapshot_in_state(state),
        detail=design_detail_in_state(state),
        task_plan=task_plan,
    )


def design_snapshot_context_json(context: dict[str, Any] | None) -> str:
    """Render a selection for a prompt, or the empty string when there is nothing to render."""
    if context is None:
        return ""
    return json.dumps(context, indent=2, sort_keys=True)


def _quoted(record: Any) -> dict[str, Any]:
    """Render one resolved frame for a prompt: what it is, then what it looks like.

    The identity first -- which file, which frame, what the author called it -- then the frame
    itself. `characters` is dropped: it is the platform's own accounting of a bound, and a
    model has no use for it.
    """
    return {
        "file_key": record.file_key,
        "node_id": record.node_id,
        "label": record.label,
        "source_url": record.source_url,
        "applies_to": list(record.applies_to),
        "design": record.content,
    }


def _clean_names(value: Any) -> Iterator[str]:
    """Yield the well-formed frame ids in a persisted list, in order, without repeats."""
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return
    yield from _unique(item.strip() for item in value if isinstance(item, str) and item.strip())


def _unique(values: Iterable[str]) -> Iterator[str]:
    """Yield each value the first time it appears."""
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            yield value


__all__ = [
    "design_detail_in_state",
    "design_node_names",
    "design_request_context",
    "design_snapshot_context",
    "design_snapshot_context_from_state",
    "design_snapshot_context_json",
    "design_snapshot_in_state",
    "scoped_design_node_ids",
]
