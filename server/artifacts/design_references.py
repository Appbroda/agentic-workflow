"""What a person can cite when they attach a design, and how it is normalized once.

A citation is a URL somebody pasted out of a browser. Figma has shipped two spellings of that
URL and two spellings of a node id, and the API accepts exactly one of each -- so the
translation happens here, once, at the boundary, and the normalized values are what everything
downstream reads. Both are recorded: the API spelling because the resolver needs it, and the
pasted URL because the console links back to what the person was actually looking at.

Verified against the live API on 2026-09-07:

* a share URL is `https://www.figma.com/design/<key>/<slug>?node-id=<a>-<b>` and, from before
  the rename, `.../file/<key>/...`. Both carry the same key.
* the API wants `<a>:<b>`. A node id sent in the URL's own `<a>-<b>` spelling is refused with
  `400 ID not-an-id is not a valid node_id`, so translating here saves a round trip that could
  only ever fail.
* file keys are `[A-Za-z0-9]`; the four real files this item was verified against have keys of
  22 and 24 characters.

Nothing here reaches the network. Every refusal below is a statement about the text somebody
pasted, which is what makes it safe to make at the request boundary -- and what keeps the
resolution in Part D the only thing that ever opens a file.

This module is a leaf: it imports nothing from this repository, so `artifacts.schemas` can hold
the citation on a persisted artifact and `storage.design_source_store` can validate an
allowlist entry against the same file-key shape, without either importing the other.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Annotated, Any, Literal, Self
from urllib.parse import parse_qs, unquote, urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator

# What a Figma file key looks like. The bound is deliberately looser than what was measured:
# Figma has never published a key format, and guessing a length would refuse a key that works.
# What it does reliably reject is the two things somebody actually pastes by mistake -- a whole
# URL, which carries `/`, `:` and `.`, and a credential, which carries `_` and `-`.
DESIGN_FILE_KEY = re.compile(r"^[A-Za-z0-9]{6,128}$")

# A node id in the API's own spelling. `1:23` for an ordinary node; `I13:13;12:57` for a node
# inside an instance, where the semicolons are the instance path. Both come back from the live
# API and both are legitimate things to cite.
_DESIGN_NODE_ID = re.compile(r"^I?\d+:\d+(?:;\d+:\d+)*$")

# The two URL path prefixes Figma has shipped for a design file. `/proto/`, `/board/` and
# `/slides/` carry a file key too and are deliberately not accepted: the extraction in Part D
# reads design nodes, and a FigJam board resolved through it would produce confident nonsense.
_DESIGN_URL_PREFIXES = ("file", "design")

_FIGMA_HOST = "figma.com"

# A branch link has at least four path segments: `design/<parentKey>/branch/<branchKey>`. The
# count matters: a file literally named "branch" produces `/design/<key>/branch` -- three
# segments, no branch key -- and is an ordinary citation of that file, not a branch link.
_BRANCH_SEGMENT_COUNT = 4

# Bounds on the text a citation carries. A request body is not a place for an unbounded string.
_MAX_URL_CHARACTERS = 2_048
_MAX_LABEL_CHARACTERS = 200

# ---------------------------------------------------------------------------------------------
# The bounds a resolved design snapshot is held to, and where the numbers come from
# ---------------------------------------------------------------------------------------------
#
# Declared here, in the leaf both the request boundary and the resolver import, so the check
# that refuses a citation at acceptance and the check that omits a node at resolution are the
# same numbers rather than two that were written to agree.
#
# Calibrated against real Figma files on 2026-09-07, the way `_SECTION_MAX_CHARACTERS` records
# -197's 16,455 characters -- a bound whose origin is not written down gets "tuned" by the next
# person into a bound that bites. What was measured, live:
#
# * `28gd2JrZO28FCN9PCKM4qK` ("OpenCRM Design": a real product file, 1,521 nodes, 369 with auto
#   layout, 68 named styles, 27 component sets). Its login screen `2:303` is **32 nodes at a
#   natural depth of 6**; its `371:42034` "Widget user profile" is **53 nodes at a natural
#   depth of 8** with 16 component instances -- the largest ordinary frame measured.
# * `VGULlnz44R0Ooe4FZKDxlhh4`, whose live capture is committed as
#   `tests/fixtures/figma/figma_file_read.json` and measured through the shipped extraction:
#   its three frames render to **8,994, 7,270 and 677 characters** -- 16,941 for the whole
#   file, 29 nodes, i.e. **584 characters per node** including the JSON nesting -- at natural
#   depths of 4, 3 and 1.
#
# At 584 characters a node, the two OpenCRM frames above render to roughly 18,700 and 31,000
# characters. `test_design_snapshot.py` asserts against the committed capture that no bound
# bites on it: the largest real frame is 22% of the per-node bound, the whole file 21% of the
# total, and the deepest is 4 of a bound of 8.
#
# Every bound below is therefore headroom on measured input rather than a trim on it, which is
# the point: a bound that bites routinely is a bound that routinely hides the design. And when
# one does bite it reports what it dropped -- see the snapshot's three omission lists.
#
# **Re-calibrated 2026-09-10 against a third file, and the sentence above is why it had to be.**
# The two files quoted above are a demo file and a design system read through a second-hand
# measurement; neither is a product screen. `EqhdoZR1UKVZsaX7jplDsP` is, and against it the
# original numbers omitted **every** frame: its ordinary screens are 119,049 to 196,447
# characters at depth 8 against a per-node bound of 40,000, so a citation of any of them
# resolved to nothing and every prompt was correctly told the design had not been shown.
#
# 584 characters a node was also low. A real product screen is ~1,470 a node at build fidelity,
# because it carries auto layout on 83 of 93 nodes, typography on every text node, and paint on
# 46 -- none of which the demo file above has any of. That is the whole reason a second capture
# is committed beside it.
#
# What replaced the two generic bounds: `MAX_DESIGN_NODE_CHARACTERS` and
# `MAX_DESIGN_TOTAL_CHARACTERS` are gone, because after 96- there is no single answer to "how
# big may a design be" -- there are two, for two renderings with two purposes. The index bounds
# and the detail bounds below say which is which, and a bound with no tier in its name would be
# the next thing somebody tuned in the wrong direction.
#
# One measurement note that cost this item a full recalibration: sizes here are `_characters`,
# which serializes with `indent=2` -- the same rendering `design_snapshot_context_json` hands a
# model. A compact measurement of the same tree is 2.6x smaller and calibrates nothing, because
# nothing ever sends a model the compact form.

# Nodes one snapshot may carry. 24 rather than a larger number because a person citing more
# than two dozen frames on one feature is describing a project; and because a whole-file
# citation resolves to a file's top-level frames, of which this is the first 24, with the rest
# reported rather than dropped.
MAX_DESIGN_NODES = 24

# How deep a cited node's children are followed **when it is indexed**. 8 is the natural depth
# of the deepest frame the first calibration measured, so it truncated nothing on either of
# those files -- but it is not the natural depth of a real product screen, which was measured at
# 17 on 2026-09-10. For the index that is fine and deliberate: a role deciding which repository
# a frame belongs to does not need the ninth level of a component's internals. For the tier an
# engineer builds from it is not fine, which is what `MAX_DESIGN_DETAIL_DEPTH` exists for.
MAX_DESIGN_DEPTH = 8

# ---------------------------------------------------------------------------------------------
# The index tier: what a frame renders to when the question is "what is this", not "build it"
# ---------------------------------------------------------------------------------------------
#
# A resolution renders every cited frame twice, for two different questions, and these are the
# bounds on the first. The index carries `path`, `name`, `type`, `component`,
# `variant_properties`, `style_names` and `text` and nothing else -- no `layout`, no
# `typography`, no `size`, no paint. It is what the product manager derives requirements from
# and what the planner assigns frames to workstreams from, because both are choosing rather
# than building. It is never what an engineer builds from: a rendering with no values in it
# produces invented hex codes that pass every gate this platform has, since no command this
# platform runs checks a colour.
#
# Measured 2026-09-10 through the shipped extraction and `_characters`, on the same real
# product file the bounds above were calibrated against:
#
# | citation                | nodes | index chars | per node | vs build fidelity |
# |-------------------------|-------|-------------|----------|-------------------|
# | `73996:19743` SECTION   |   126 |      57,852 |      459 |  71% smaller      |
# | `73996:19746` FRAME     |    93 |      41,934 |      450 |  69% smaller      |
# | `74046:28253` FRAME     |    36 |      12,028 |      334 |  66% smaller      |
#
# ~450 characters a node, so a frame indexes at roughly a third of what it costs to build. The
# saving is real and per-screen; what it is *not* is a licence to carry thirty screens for the
# price of one -- a 30-screen flow indexes to about 1.26M characters, which is nine times one
# screen at build fidelity. 96- corrects an earlier draft that put this at 221 characters a
# node; that number was a compact serialization, and `_characters` renders `indent=2` because
# `indent=2` is what `design_snapshot_context_json` actually hands a model.

# One node's rendered text in index mode. 150,000 is headroom on the largest measured index
# (57,852, the 126-node section) by a factor of 2.6, per the rule the bounds above are written
# to: a bound that bites routinely is a bound that routinely hides the design. A first draft
# of this said 30,000, which would have omitted both large frames measured -- i.e. re-created
# the exact bug the index tier exists to fix, one tier further down.
MAX_DESIGN_INDEX_NODE_CHARACTERS = 150_000

# The whole index. A runaway guard, not a selection bound: at 450 characters a node this holds
# roughly 3,300 nodes, which is a thirty-screen flow at the measured 93 nodes a screen with
# room to spare. A frame past it is reported as `index_total_exhausted` rather than dropped,
# because a frame missing from the index is a frame hidden from the planner -- the role that
# decides which repository it belongs to -- and that is the failure this whole item is about.
#
# Sized to hold a whole flow even though one feature cannot *build* one: this platform enforces
# one workstream per repository, so thirty frontend screens are one workstream and one detail
# bound however the index is sized. What a complete index buys is the decomposition -- reading
# the whole flow in order to split it across several features, which is 97-. Bounding the index
# to what one feature can build would remove the only tier that ever sees the flow whole.
MAX_DESIGN_INDEX_TOTAL_CHARACTERS = 1_500_000

# ---------------------------------------------------------------------------------------------
# The detail tier: what one repository's workstream is actually handed to build from
# ---------------------------------------------------------------------------------------------
#
# Per REPOSITORY, which on this platform is also per workstream and per child workflow: the
# execution plan refuses two workstreams naming one repository
# (`RepositoryExecutionPlanArtifact.workstream_references_must_be_valid`) and the child id is
# keyed `feature_id:repository_id`. That 1:1 is why these are not "per feature" bounds and also
# why they buy less than 96- first claimed: thirty frontend screens are one repository and
# therefore one of these bounds, however it is sized. Nothing here makes a thirty-screen flow
# buildable in one feature. What it does is stop one *screen* being omitted whole, which is
# what happens today -- every top-level frame on the measured file is over the old 40,000.
#
# Measured 2026-09-10 through the shipped extraction and `_characters`, at the detail depth
# below:
#
# | citation                | nodes @10 | `_characters` @10 |
# |-------------------------|-----------|-------------------|
# | `73996:19743` SECTION   |       247 |           392,810 |
# | `73996:19746` FRAME     |       166 |           276,188 |
# | `74046:28754` FRAME     |       154 |           255,756 |
#
# The old numbers these replace were 40,000 per node and 80,000 in total, calibrated against a
# demo file whose largest frame is 8,994 characters. Against a real product screen they omit
# everything.

# How deep a cited node's children are followed when it is resolved for building. 10, not the
# 8 the index uses and not the 24 an earlier draft of 96- proposed.
#
# 8 is too shallow: it stops at 93 of the frame's 411 real nodes, and a depth-8 view of a
# depth-17 screen is not a screen anybody can implement. Uncapped is too deep for a different
# reason -- the same frame is 765,949 characters at its full depth of 17, about 190,000 tokens,
# and `budget-fits-the-model` would then spend an engineer's whole context on one screen and
# starve the source it has to change. Depth 10 doubles the nodes depth 8 gives (166) at 276,188
# characters, which leaves room for the repository snapshot beside it.
#
# The cut is still *reported*, per node, as `children_beyond_depth_bound` -- so a frame shown to
# depth 10 says so, rather than reading as a frame whose children end there.
#
# This is now a *ceiling* rather than the depth every frame renders at.
# `FigmaDesignResolver._deepest_that_fits` starts here and falls back two levels at a time until
# the rendering fits `MAX_DESIGN_DETAIL_NODE_CHARACTERS`, so a small frame gets the extra
# fidelity and a large one lands back on 10 instead of being omitted. Measured on
# AB-Feature-229's login frame: depth 10 is 276,188 characters over 166 nodes with 30 subtrees
# cut, depth 12 is 573,536 over 312, depth 14 is 754,605 over 404 with only 2 cuts left, and the
# frame is complete at depth 17 and 765,949. That frame therefore still renders at 10; a frame
# half its size now renders at 14.
MAX_DESIGN_DETAIL_DEPTH = 14

# How deep the *index tail* of a detail rendering reaches below `MAX_DESIGN_DETAIL_DEPTH`.
# 24 is past the deepest real frame measured (17), so in practice this means "to the leaves".
#
# This is what stops a screen arriving as correctly-positioned empty boxes. Measured on
# AB-Feature-231's login frame: a flat build rendering at depth 10 carries 24 of the design's
# 71 text strings, and the implementation it produced drew every container and left it blank,
# because the words inside were below the cut. Flat build at depth 14 carries all 71 -- and
# costs 754,605 characters, about 188,000 tokens against a 272,000-token window, which would
# starve the repository context the change has to be written against.
#
# The tail is index fidelity: names, text and style names, no paint and no geometry. At ~450
# characters a node against ~1,470, the hybrid carries all 411 nodes and all 71 strings for
# 433,465 characters -- inside the existing per-node bound, with no bound raised.
MAX_DESIGN_DETAIL_INDEX_DEPTH = 24

# One node's rendered text at build fidelity, subtree included. 450,000 fits the largest single
# citation measured at depth 10 -- the 247-node SECTION at 392,810 -- with 15% headroom.
#
# This is the budget `_deepest_that_fits` spends: a frame too large at `MAX_DESIGN_DETAIL_DEPTH`
# is re-rendered shallower until it fits, and only one that will not fit even at depth 10 is
# omitted whole and named. The omission is still whole rather than partial, for the reason it
# always was: a judge shown two thirds of a frame reads a layout that is missing children and
# finds a violation that is not there. What changed is that a frame now has to be enormous to
# earn that, instead of merely being one level deeper than the bound.
#
# 380,000 is roughly 95,000 tokens: about 35% of the 272,000-token window the current coding
# model declares. The repository snapshot already takes 20% of that window by design
# (`repository_snapshot_budget`), and the remainder has to hold the instructions, the task
# plan, the contract, a prior review, the previous attempt's diff and the model's own
# reasoning output.
#
# This number is why AB-Feature-235 died. At 520,000 the login frame rendered 462,990
# characters -- 116,000 tokens -- *outside* the snapshot's accounting entirely, so context
# alone reached 170,000 tokens of 272,000. The initial implementation fit; the review
# remediation, which adds the findings and the previous diff on top, did not, and five
# consecutive provider errors spent the workstream's fault allowance. AB-Feature-229 had
# succeeded with a 69,000-token design.
#
# It should be derived from the routed model's declared window rather than fixed here, the
# way the snapshot budget is. That is real work -- the resolver would have to learn the
# routing -- and this constant is the honest interim: correct for the model this deployment
# runs, and wrong the day it runs a smaller one.
#
# Raised from 450,000 when every rendered node started carrying its Figma id. The ids are what
# the asset export addresses images by -- without them it requested `render_preview(key, "")`
# and Figma answered 400 six times -- and they cost the login frame 29,525 characters, taking
# its hybrid rendering from 433,465 to 462,990. At the old bound that frame was over, and at
# the fallback floor there is nowhere shallower to go, so it would have been omitted whole:
# adding node ids would have deleted the design. 520,000 leaves headroom and stays under
# `MAX_DESIGN_DETAIL_TOTAL_CHARACTERS`, so a second large frame in one workstream is still
# reported as omitted rather than silently doubling an engineer's prompt.
MAX_DESIGN_DETAIL_NODE_CHARACTERS = 380_000

# Everything one repository's workstream is handed. 560,000 holds the two ordinary product
# screens measured at depth 10 (531,944 together), which is what a workstream realistically
# builds. A third is omitted and reported rather than trimmed.
#
# This is the number to re-check against Risk 2 before raising: two screens is already ~140,000
# tokens of design in one engineer prompt, competing with the source context that
# `required-context-omitted-silently` was written about. If it starves, the remedy is fewer
# screens per feature -- several features, one per screen group, which is 97- -- and never a
# trimmed rendering. Trimming is what produces a judge who finds a violation that is not there.
MAX_DESIGN_DETAIL_TOTAL_CHARACTERS = 560_000

# How many nodes one *citation* may name. The snapshot bound is the one that matters; this is
# the request-boundary form of it, so a single pasted URL cannot ask for more than a snapshot
# could ever hold.
_MAX_NODE_IDS = MAX_DESIGN_NODES


# Which of the two questions a rendering answers. `build` is what an engineer implements from
# and the only mode any judge may be handed; `index` is what the product manager and the
# planner *decide* from. See `extract_design_node`.
type DesignRenderMode = Literal["index", "build"]

# Every key an index-mode rendering may carry. Declared rather than left implicit in `_render`
# because two readers need it: the tests that assert no value key survives at any depth, and
# `design_record_is_buildable` below, which is what stops a valueless record reaching a model
# that would fill the gaps in with invented hex codes.
DESIGN_INDEX_KEYS = frozenset(
    {
        "node_id",
        "path",
        "name",
        "type",
        "component",
        "variant_properties",
        "style_names",
        "text",
        "children",
        "children_beyond_depth_bound",
    }
)

# The keys that only exist at build fidelity. A record carrying none of them is an index
# record, whatever produced it.
DESIGN_BUILD_ONLY_KEYS = frozenset(
    {
        "size",
        "layout",
        "constraints",
        "typography",
        "fills",
        "strokes",
        "stroke_weight",
        "effects",
        "corner_radius",
        "corner_radii",
    }
)


def design_content_is_buildable(content: Mapping[str, Any]) -> bool:
    """Whether one rendered frame carries the values an engineer would need to implement it.

    The guard behind Risk 1 of 96-, and the reason it is a predicate rather than a convention:
    an index record handed to the Engineer, its self-review or the Reviewer reads as a design
    with no colours, no type and no spacing in it, and a model asked to build that invents all
    three. Invented values then pass every gate this platform has, because no command it runs
    checks a colour -- so the failure is silent, ships, and is only visible to a person looking
    at the screen.

    True when any node in the tree carries a build-only key. Asked of the whole tree and not
    just the root, because a frame's own root may legitimately carry nothing but a name while
    its children carry the paint -- and a record whose *every* node is value-free is an index
    record whatever produced it, which is exactly what this refuses.

    The residual edge is a build-mode render of a subtree that genuinely declares no geometry,
    no paint, no layout and no type anywhere. On the real files measured that does not occur --
    `absoluteBoundingBox` and `constraints` are present on every node of the frames measured --
    and treating it as not-buildable is the safe direction: it costs an honest omission, where
    the other direction costs invented values nobody catches.
    """
    if DESIGN_BUILD_ONLY_KEYS & content.keys():
        return True
    return any(
        isinstance(child, Mapping) and design_content_is_buildable(child)
        for child in (content.get("children") or [])
    )


class DesignReferenceError(ValueError):
    """Raised when a pasted design URL cannot be read as a citation."""


class DesignReference(BaseModel):
    """One design somebody attached to a feature request, normalized once at the boundary.

    `url` is what they pasted, kept verbatim so the console can link back to it. `file_key`
    and `node_ids` are derived from it, and derived here rather than by each reader: two
    readers translating a URL is two chances to disagree about which frame was meant.

    An empty `node_ids` is a whole-file citation and is legitimate. It resolves to the file's
    top-level frames, bounded, with anything past the bound reported rather than dropped --
    see Part D's omission lists. It is not a way to ask for "everything".
    """

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    url: Annotated[str, Field(min_length=1, max_length=_MAX_URL_CHARACTERS)]
    # Defaulted so `DesignReference(url=...)` is the ordinary construction: the before-validator
    # below derives this from the URL, and the after-validator refuses an empty or malformed one
    # with a sentence rather than a schema error. A required field here would have made every
    # caller -- the request boundary, the resolver, every test -- restate the derivation.
    file_key: Annotated[str, Field(max_length=128)] = ""
    node_ids: Annotated[list[str], Field(max_length=_MAX_NODE_IDS)] = Field(default_factory=list)
    # What this frame is, in the author's words: "empty state", "mobile breakpoint". Optional,
    # because a person pasting one URL should not have to name it.
    label: Annotated[str, Field(max_length=_MAX_LABEL_CHARACTERS)] = ""
    # Which repositories this design applies to. Empty means the planner decides, which is the
    # right default: this platform does not know which repository renders UI, and `role` is a
    # descriptive label that must never be used to guess.
    applies_to: list[Annotated[str, Field(min_length=1, max_length=128)]] = Field(
        default_factory=list
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_the_pasted_url(cls, data: Any) -> Any:
        """Derive `file_key` and `node_ids` from the URL when the caller did not supply them.

        A `mode="before"` validator rather than a field validator because it writes two fields
        from a third, and it is idempotent: a persisted citation already carrying its derived
        values re-derives the same ones, so every artifact written before or after this reads
        the same way.

        Supplied values are *checked against* the URL rather than trusted over it. Every value
        this platform ever stored was derived from the same URL, so a stored artifact reloads
        untouched -- but a wire caller supplying `file_key` or `node_ids` that disagree with
        the URL used to win silently, and everything downstream (the resolver, the console's
        link back, the duplicate check) would then be about two different designs at once.
        Malformed supplied values are deliberately left for the after-validator, which names
        them; only a well-formed disagreement is a mismatch.
        """
        if not isinstance(data, dict):
            return data
        raw_url = data.get("url")
        if not isinstance(raw_url, str) or not raw_url.strip():
            return data
        parsed = parse_design_url(raw_url)
        normalized = dict(data)
        supplied_key = normalized.get("file_key")
        if (
            isinstance(supplied_key, str)
            and DESIGN_FILE_KEY.fullmatch(supplied_key)
            and supplied_key != parsed.file_key
        ):
            msg = (
                f"this citation's file_key '{supplied_key}' does not match its URL, which "
                f"names the file '{parsed.file_key}'"
            )
            raise DesignReferenceError(msg)
        normalized.setdefault("file_key", parsed.file_key)
        supplied_nodes = normalized.get("node_ids")
        if not supplied_nodes:
            normalized["node_ids"] = list(parsed.node_ids)
        elif isinstance(supplied_nodes, list) and all(
            isinstance(item, str) for item in supplied_nodes
        ):
            # The URL's own dash spelling is accepted here exactly as it is in the URL, so a
            # caller restating the same frames in either spelling is a match, not a mismatch.
            restated = [str(item).strip().replace("-", ":") for item in supplied_nodes]
            if all(_DESIGN_NODE_ID.fullmatch(item) for item in restated):
                if set(restated) != set(parsed.node_ids):
                    msg = (
                        f"this citation's node_ids {sorted(set(restated))} do not match its "
                        f"URL, which names {sorted(set(parsed.node_ids))}"
                    )
                    raise DesignReferenceError(msg)
                normalized["node_ids"] = restated
        return normalized

    @model_validator(mode="after")
    def derived_values_must_be_well_formed(self) -> Self:
        """Refuse a citation whose derived values could not be resolved or read back.

        Checked after normalization as well as during it, because a caller may supply
        `file_key` and `node_ids` directly -- a persisted artifact does -- and a stored value
        that no longer parses is something to find at load rather than at a fetch.
        """
        if not DESIGN_FILE_KEY.fullmatch(self.file_key):
            msg = f"'{self.file_key}' is not a Figma file key"
            raise DesignReferenceError(msg)
        for node_id in self.node_ids:
            if not _DESIGN_NODE_ID.fullmatch(node_id):
                msg = (
                    f"'{node_id}' is not a Figma node id. The API's spelling is '1:23'; a URL "
                    "writes the same id as '1-23', and this platform translates it"
                )
                raise DesignReferenceError(msg)
        if len(set(self.node_ids)) != len(self.node_ids):
            msg = "a citation must not name the same node twice"
            raise DesignReferenceError(msg)
        if len(set(self.applies_to)) != len(self.applies_to):
            msg = "a citation must not name the same repository twice"
            raise DesignReferenceError(msg)
        return self

    @property
    def cites_whole_file(self) -> bool:
        """Whether this citation names no node, and so asks for the file's top-level frames."""
        return not self.node_ids

    @property
    def pairs(self) -> tuple[tuple[str, str], ...]:
        """The `(file_key, node_id)` pairs this citation claims, for duplicate detection.

        A whole-file citation claims the pair `(file_key, "")`, so citing a whole file twice is
        as ambiguous as citing one frame twice and is refused the same way.
        """
        if self.cites_whole_file:
            return ((self.file_key, ""),)
        return tuple((self.file_key, node_id) for node_id in self.node_ids)


class ParsedDesignUrl(BaseModel):
    """What one pasted URL says: which file, and which nodes within it."""

    model_config = ConfigDict(frozen=True)

    file_key: str
    node_ids: tuple[str, ...]


def parse_design_url(value: str) -> ParsedDesignUrl:
    """Read a pasted Figma URL into a file key and the nodes it names, or refuse it.

    Refusals mirror `RepositorySpec.repository_url_must_not_embed_credentials` in shape and in
    reason: a URL is a thing a person types, so being told which part of it is wrong beats
    reading a 422 that names a JSON path. The client mirrors these rules for the same reason.
    """
    raw = value.strip()
    try:
        parsed = urlsplit(raw)
    except ValueError as error:
        msg = "that does not look like a URL"
        raise DesignReferenceError(msg) from error
    if parsed.scheme not in ("http", "https"):
        msg = "a design URL must be an HTTP(S) URL"
        raise DesignReferenceError(msg)
    if parsed.username or parsed.password:
        msg = "do not put credentials in a design URL"
        raise DesignReferenceError(msg)
    host = (parsed.hostname or "").lower()
    if host != _FIGMA_HOST and not host.endswith(f".{_FIGMA_HOST}"):
        msg = f"a design URL must be on {_FIGMA_HOST}; this one is on '{host or 'no host'}'"
        raise DesignReferenceError(msg)
    segments = [segment for segment in parsed.path.split("/") if segment]
    if len(segments) < 2 or segments[0] not in _DESIGN_URL_PREFIXES:
        msg = (
            "that Figma link does not name a design file. A design file's URL is "
            "figma.com/design/<key>/... (or figma.com/file/<key>/... from before the rename); "
            "a prototype, FigJam or Slides link is a different kind of document"
        )
        raise DesignReferenceError(msg)
    file_key = segments[1]
    if not DESIGN_FILE_KEY.fullmatch(file_key):
        msg = f"'{file_key}' is not a Figma file key"
        raise DesignReferenceError(msg)
    if len(segments) >= _BRANCH_SEGMENT_COUNT and segments[2] == "branch":
        # A branch link is `/design/<parentKey>/branch/<branchKey>/<Name>`. Reading segment 1
        # as the file key -- which is what every non-branch URL means -- would silently
        # snapshot the *parent* file: a design that looks resolved and is not the one the
        # person was looking at. Refused by name until branch resolution (fetching by the
        # branch key) is built and verified against the live API.
        msg = (
            "this is a branch link; branches are not supported yet. Paste a link to the "
            "main file, or merge the branch first"
        )
        raise DesignReferenceError(msg)
    return ParsedDesignUrl(file_key=file_key, node_ids=_node_ids_in(parsed.query))


def _node_ids_in(query: str) -> tuple[str, ...]:
    """Read every node id a share URL's `node-id` names, in the API's own spelling.

    `parse_qs` has already percent-decoded, so `node-id=1%3A23` arrives as `1:23` and
    `node-id=1-23` as `1-23`. The dash spelling is the browser's; the colon is the API's, and
    a node id contains no other dash, so one replacement translates both forms and leaves an
    instance path (`I13:13;12:57`) intact.
    """
    values = parse_qs(query).get("node-id") or []
    node_ids: list[str] = []
    for value in values:
        for candidate in unquote(value).split(","):
            node_id = candidate.strip().replace("-", ":")
            if node_id and node_id not in node_ids:
                node_ids.append(node_id)
    return tuple(node_ids)


def duplicate_design_pairs(references: list[DesignReference]) -> list[str]:
    """Return the `(file_key, node)` pairs cited more than once, readably.

    Refused for the reason a repeated story or requirement id is refused: the citation is
    ambiguous, and the ambiguity would be resolved silently -- one of the two labels would win
    and the other would vanish from the snapshot the engineer is judged against.
    """
    seen: set[tuple[str, str]] = set()
    duplicates: list[str] = []
    for reference in references:
        for pair in reference.pairs:
            if pair in seen:
                readable = f"{pair[0]} (whole file)" if not pair[1] else f"{pair[0]}#{pair[1]}"
                if readable not in duplicates:
                    duplicates.append(readable)
            seen.add(pair)
    return duplicates


__all__ = [
    "DESIGN_BUILD_ONLY_KEYS",
    "DESIGN_FILE_KEY",
    "DESIGN_INDEX_KEYS",
    "MAX_DESIGN_DEPTH",
    "MAX_DESIGN_DETAIL_DEPTH",
    "MAX_DESIGN_DETAIL_NODE_CHARACTERS",
    "MAX_DESIGN_DETAIL_TOTAL_CHARACTERS",
    "MAX_DESIGN_INDEX_NODE_CHARACTERS",
    "MAX_DESIGN_INDEX_TOTAL_CHARACTERS",
    "MAX_DESIGN_NODES",
    "DesignReference",
    "DesignReferenceError",
    "DesignRenderMode",
    "ParsedDesignUrl",
    "design_content_is_buildable",
    "duplicate_design_pairs",
    "parse_design_url",
]
