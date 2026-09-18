"""The one deterministic thing this platform can say about "does it match the design"."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from tools.design_value_conformance import (
    added_change_text,
    design_value_issues,
    design_values,
)

FIXTURES = Path(__file__).parent / "fixtures" / "figma"


def _frame() -> dict[str, Any]:
    """The committed product frame, rendered the way the Engineer is handed it."""
    from adapters.figma_adapter import FigmaNodeSubtree
    from services.design_resolution import extract_design_node

    payload: dict[str, Any] = json.loads((FIXTURES / "figma_product_frame.json").read_text())
    entry = payload["nodes"]["73996:19746"]
    subtree = FigmaNodeSubtree(
        node_id="73996:19746",
        document=entry["document"],
        styles=entry.get("styles") or {},
        components=entry.get("components") or {},
        component_sets=entry.get("componentSets") or {},
    )
    content, _, _ = extract_design_node(subtree)
    return content


def test_a_real_frame_names_colours_and_families_worth_checking() -> None:
    """Read off the committed capture, so the values are a real design's and not invented."""
    colours, families = design_values(_frame())

    assert colours, "a product screen with no distinctive colour is not a product screen"
    assert all(value.startswith("#") for value in colours)
    assert all(value == value.lower() for value in colours), "colours compare case-insensitively"
    # Pure black and white are in every design and every stylesheet, so they identify nothing.
    assert "#000000" not in colours
    assert "#ffffff" not in colours


def test_a_change_that_uses_the_designs_palette_has_nothing_reported() -> None:
    """The empty result is the one that matters: a silent check is a check nobody disables."""
    colours, families = design_values(_frame())
    used = "\n".join([*sorted(colours), *sorted(families)])

    assert design_value_issues([_frame()], changed_text=used) == ()


def test_a_change_that_ignores_the_design_is_told_which_values_it_never_used() -> None:
    """The finding names a bounded sample and counts the rest, rather than listing forty."""
    issues = design_value_issues([_frame()], changed_text="body { color: rebeccapurple; }")

    assert issues, "a change sharing no value with the design must not read as conforming"
    joined = " ".join(issues)
    assert "Absent:" in joined
    assert "and this change mentions 0 of them" in joined
    # It says what a correct absence looks like, because a design-system token is a correct
    # absence and a check that called it a defect would be trained away.
    assert "use the token" in joined


def test_a_colour_written_in_any_stylesheet_spelling_counts_as_used() -> None:
    """`#0A0A0A`, `#0a0a0a` and the shorthand `#aaa` are not three different colours.

    A check that reported a correctly written shorthand as absent would teach its reader to
    ignore every line it produces, which is worse than not running at all.
    """
    frame = {"fills": [{"type": "SOLID", "hex": "#aabbcc"}], "children": []}
    shorthand = {"fills": [{"type": "SOLID", "hex": "#aaaaaa"}], "children": []}

    assert design_value_issues([frame], changed_text="color: #AABBCC;") == ()
    assert design_value_issues([shorthand], changed_text="color: #aaa;") == ()
    assert design_value_issues([frame], changed_text="color: #aabbcd;") != ()


def test_a_feature_that_cites_nothing_distinctive_says_nothing() -> None:
    """Byte-identical prompts for the features that cite no design, which is most of them."""
    assert design_value_issues([], changed_text="anything") == ()
    assert design_value_issues([{"fills": [{"hex": "#ffffff"}]}], changed_text="") == ()


def test_only_the_change_counts_and_not_the_repository_it_landed_in() -> None:
    """A colour already in the stylesheet is not evidence this change used it.

    Counting the whole file would let an attempt that restyled nothing report full coverage,
    which is precisely the claim this exists to refuse.
    """
    frame = {"fills": [{"type": "SOLID", "hex": "#123456"}], "children": []}

    assert design_value_issues([frame], changed_text="") != ()


@pytest.mark.asyncio
async def test_the_inspection_never_writes_the_workspace_it_inspects() -> None:
    """Read-only is load-bearing: the commit gate stages in the same checkout.

    An earlier draft ran `git add --intent-to-add --all` to make untracked files visible to
    `git diff`. That mutates the index, and the Engineer's commit gate stages and commits in
    the very workspace this inspects -- so an inspection could have changed what a later
    commit picked up. An inspection that can alter the change it is inspecting is worse than
    no inspection at all.

    Asserted on the commands actually issued, because that is the property: any future
    implementation that reaches for a mutating porcelain command fails here.
    """
    issued: list[tuple[str, ...]] = []

    class _RecordingRunner:
        async def run(
            self,
            command: Sequence[str],
            cwd: Path,
            timeout_seconds: float,
            cancellation_token: Any,
            environment: Any = None,
        ) -> Any:
            del cwd, timeout_seconds, cancellation_token, environment
            issued.append(tuple(command))
            return SimpleNamespace(return_code=0, stdout="", stderr="")

    from services.cancellation import MockCancellationToken

    await added_change_text(
        Path("."),
        runner=cast(Any, _RecordingRunner()),
        timeout=1.0,
        cancellation_token=MockCancellationToken(),
    )

    assert issued, "the inspection ran no command at all"
    mutating = {"add", "stash", "checkout", "reset", "commit", "restore", "clean", "rm", "mv"}
    for command in issued:
        assert command[0] == "git"
        assert command[1] not in mutating, f"{command} writes the workspace it inspects"


def test_a_design_finding_is_told_and_never_repaired() -> None:
    """The difference between this check and the two it sits beside, as a property.

    A wiring or placement finding names something definitely wrong and earns a repair pass.
    A design-value finding may name something correct -- a repository with a design system
    should write its token rather than the literal, which reads here as the colour being
    absent -- so it is reported and never repaired.

    This is not a preference. AB-Feature-231 and AB-Feature-232 both had clean wiring and
    placement, so this check alone started a remediation loop, and five consecutive provider
    faults inside that loop spent each workstream's entire fault allowance. AB-Feature-229,
    which ran before the check existed, completed.
    """
    from agents.engineer.agent import _InAttemptFindings

    design_only = _InAttemptFindings(design_issues=("the design specifies 14 colours",))
    assert design_only.issues == (), "a design finding must not start a repair pass"
    assert design_only.diagnostics, "and must still be told to whoever reads the attempt"

    # A real defect beside it still earns its pass, and the design finding rides along in the
    # same diagnostic set rather than being dropped.
    with_wiring = _InAttemptFindings(
        wiring_issues=("nothing reaches the new module",),
        design_issues=("the design specifies 14 colours",),
    )
    assert with_wiring.issues == ("nothing reaches the new module",)
    assert len(with_wiring.diagnostics) == 2


def test_placed_artwork_never_triggers_the_required_context_refusal() -> None:
    """A PNG can never be quoted in a text snapshot, so demanding it refuses every design.

    The required-context guard is right in general: an attempt that cannot see the file it was
    told to change answers findings with guesses. But a placed design asset is an input
    referenced *by path* -- the Engineer moves the file and points at it, never reads its
    bytes -- and it is dropped as `unreadable` every time by construction.

    AB-Feature-234 was refused before its first model call over three exported images, having
    placed all six correctly.
    """
    import pytest as _pytest

    from agents.engineer.agent import (
        _PLACED_ASSET_DIRECTORY,
        RequiredContextRefusal,
        _refuse_if_required_context_dropped,
    )
    from services.design_resolution import _ASSET_DIRECTORY

    # The two constants are duplicated across a dependency boundary; they must agree.
    assert _PLACED_ASSET_DIRECTORY == _ASSET_DIRECTORY

    asset = f"{_ASSET_DIRECTORY}/74046-28182-image-41.png"
    context = {
        "required_dropped_paths": [asset],
        "required_dropped_reasons": {asset: "unreadable"},
    }
    # Named by the plan, dropped as unreadable -- and not a refusal.
    _refuse_if_required_context_dropped(context, assigned_paths=[asset], diagnostic_file_paths=[])

    # The same image under the directory this repository actually keeps images in. Keying the
    # exclusion on `.design/assets` is what let AB-Feature-241 die on exactly the files
    # AB-Feature-234 had died on, once `fix/108` moved them to `public/`: a rule about what a
    # file *is* survives a change to where it goes, a rule about where it lives does not.
    served = "public/74046-28182-image-41.png"
    _refuse_if_required_context_dropped(
        {
            "required_dropped_paths": [served],
            "required_dropped_reasons": {served: "unreadable"},
        },
        assigned_paths=[served],
        diagnostic_file_paths=[],
    )

    # An image dropped because a *budget* ran out is a different fact and still refuses:
    # the bytes were usable, the snapshot simply could not afford them.
    with _pytest.raises(RequiredContextRefusal):
        _refuse_if_required_context_dropped(
            {
                "required_dropped_paths": [served],
                "required_dropped_reasons": {served: "required_budget_exhausted"},
            },
            assigned_paths=[served],
            diagnostic_file_paths=[],
        )

    # A real source file dropped the same way still refuses; the exclusion is narrow.
    source = "src/components/Authentication/sign-in.tsx"
    with _pytest.raises(RequiredContextRefusal):
        _refuse_if_required_context_dropped(
            {
                "required_dropped_paths": [source],
                "required_dropped_reasons": {source: "unreadable"},
            },
            assigned_paths=[source],
            diagnostic_file_paths=[],
        )
