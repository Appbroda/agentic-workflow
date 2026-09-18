"""The deterministic placement check, and the three conditions that have to hold together.

49- Part D. Run 183's frontend was rejected on attempt 0 because its scoped acceptance
criteria named the exact file -- `bulkCreateApps` exported from
`src/apiUtils/allapps.apiUtils.js`, "declared alongside the existing route constants in that
module" -- and the implementation wrote a sibling module in that directory instead. A whole
attempt to state a fact `git status` already knew.

This is the one check in this family that can be wrong about a legitimate layout, so most of
the propositions here are **negative**: the shapes on which it must stay silent. Each one is a
false positive that would cost a real repair pass to disprove, and any of them firing is the
evidence for dropping the check, which is why they are asserted rather than assumed.

`realrepo` for everything that reads a checkout, because the whole check is `git status` and
`git ls-files` over real bytes and a doubled runner would prove only that the code calls what
it calls. The end-to-end proof -- the diagnostic reaching the Engineer's repair pass inside
one attempt -- is D1 in `test_real_repository_gates.py`.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import pytest

from services.cancellation import MockCancellationToken
from services.process_runner import AsyncioProcessRunner
from tools.assigned_file_conformance import (
    ASSIGNED_FILE_DIAGNOSTIC_PREFIX,
    NullAssignedFileChecker,
    RepositoryAssignedFileChecker,
    assigned_file_conformance_issues,
    assigned_file_diagnostic,
    is_assigned_file_diagnostic,
)
from tools.reachability import reachability_diagnostic

# The plan's assignment, spelled as 183's plan spelled it: an exact path to a file that
# already exists, alongside the area strings such a list usually also carries.
_ASSIGNED = "src/apiUtils/allapps.apiUtils.js"
_PLAN = ("src", "src/apiUtils", _ASSIGNED)


# --------------------------------------------------------------------------------------
# What the Engineer's repair loop is allowed to see
# --------------------------------------------------------------------------------------


def test_the_tag_says_which_check_spoke_and_the_wiring_tag_is_not_this_one() -> None:
    """The two checks give contradictory instructions, so the repair must tell them apart.

    A wiring finding says the added module is finished and a *host* needs editing; this one
    says the added module is in the wrong place. A repair handed the wrong sentence looks for
    the defect in the wrong file, which is why the predicate is a tag and not a guess at the
    wording.
    """
    issue = f"This workstream's plan assigns {_ASSIGNED}, a file that already exists."

    tagged = assigned_file_diagnostic(issue)

    assert is_assigned_file_diagnostic(tagged)
    assert tagged.startswith(ASSIGNED_FILE_DIAGNOSTIC_PREFIX)
    assert issue in tagged
    assert not is_assigned_file_diagnostic(issue)
    assert not is_assigned_file_diagnostic(reachability_diagnostic(issue))


@pytest.mark.asyncio
async def test_a_composition_with_no_checker_inspects_nothing(tmp_path: Path) -> None:
    """A deployment or a double that wires no checker behaves exactly as before this part."""
    assert await NullAssignedFileChecker().issues(tmp_path, assigned_paths=_PLAN) == ()


# --------------------------------------------------------------------------------------
# The three conditions, against real checkouts
# --------------------------------------------------------------------------------------


def _git(root: Path, *arguments: str) -> None:
    """Run one git command for real, because every answer here comes out of a checkout."""
    subprocess.run(("git", *arguments), cwd=root, check=True, capture_output=True, timeout=30)


def _checkout(root: Path) -> Path:
    """A checkout holding the file 183's plan named, and a neighbour beside it.

    Committed, because the whole check is about what an *uncommitted* change did to a file
    the checkout already held.
    """
    (root / "src" / "apiUtils").mkdir(parents=True)
    (root / "src" / "pages").mkdir(parents=True)
    (root / "src" / "apiUtils" / "allapps.apiUtils.js").write_text(
        "export const ALL_APPS = '/apps';\nexport const APP_DETAIL = '/apps/:id';\n",
        encoding="utf-8",
    )
    (root / "src" / "apiUtils" / "home.apiUtils.js").write_text(
        "export const HOME = '/';\n", encoding="utf-8"
    )
    (root / "src" / "pages" / "AllApps.js").write_text(
        "export default function AllApps() { return null; }\n", encoding="utf-8"
    )
    _git(root, "init", "--quiet")
    _git(root, "config", "user.email", "engineer@example.test")
    _git(root, "config", "user.name", "Engineer")
    _git(root, "add", "-A")
    _git(root, "commit", "--quiet", "-m", "The module the plan assigns")
    return root


async def _inspect(root: Path, *, assigned_paths: tuple[str, ...] = _PLAN) -> tuple[str, ...]:
    """Ask the inspection about this checkout, exactly as the Engineer's seam asks it."""
    return await assigned_file_conformance_issues(
        root,
        runner=AsyncioProcessRunner(),
        timeout=60.0,
        cancellation_token=MockCancellationToken(),
        assigned_paths=assigned_paths,
    )


@pytest.mark.realrepo
@pytest.mark.asyncio
async def test_a_sibling_module_beside_an_untouched_assigned_file_is_reported(
    tmp_path: Path,
) -> None:
    """All three conditions, which is the only shape that fires: 183's frontend attempt 0.

    The finding has to name **both** paths. A diagnostic saying only "this is in the wrong
    place" is a rejection the pass cannot act on; the plan's file and the module written
    beside it are the two facts that make it one edit.
    """
    root = _checkout(tmp_path / "repo")
    (root / "src" / "apiUtils" / "bulkCreate.apiUtils.js").write_text(
        "export const BULK_CREATE_APPS = '/apps/bulk';\n", encoding="utf-8"
    )

    issues = await _inspect(root)

    assert len(issues) == 1, issues
    assert _ASSIGNED in issues[0]
    assert "src/apiUtils/bulkCreate.apiUtils.js" in issues[0]
    # The instruction is the plan's file, as an edit, and an escape hatch that costs nothing:
    # this check can be wrong, and an attempt that explains itself must not be stuck.
    assert "`edits` entry" in issues[0]
    assert "completion summary" in issues[0]


@pytest.mark.realrepo
@pytest.mark.asyncio
async def test_a_change_that_touches_the_assigned_file_is_silent(tmp_path: Path) -> None:
    """The first negative. Condition 2 fails, so nothing is wrong here at all.

    The change did exactly what the plan asked -- it declared the new constant in the module
    the plan named -- and *also* added a module beside it, which is ordinary. Firing here
    would make every change that both edits its assigned file and adds a helper pay a repair
    pass.
    """
    root = _checkout(tmp_path / "repo")
    (root / "src" / "apiUtils" / "allapps.apiUtils.js").write_text(
        "export const ALL_APPS = '/apps';\n"
        "export const APP_DETAIL = '/apps/:id';\n"
        "export const BULK_CREATE_APPS = '/apps/bulk';\n",
        encoding="utf-8",
    )
    (root / "src" / "apiUtils" / "bulkCreate.apiUtils.js").write_text(
        "export const bulkCreatePayload = () => ({});\n", encoding="utf-8"
    )

    assert await _inspect(root) == ()


@pytest.mark.realrepo
@pytest.mark.asyncio
async def test_a_change_that_adds_nothing_in_that_directory_is_silent(tmp_path: Path) -> None:
    """The second negative, and the one the spec insists on: condition 2 alone is routine.

    A plan names files a change legitimately does not modify all the time -- context, a
    consumer, a file the requirement merely mentions. What made 183 wrong was the *sibling
    creation*, so an addition somewhere else in the checkout is not this defect and must not
    be reported as it.
    """
    root = _checkout(tmp_path / "repo")
    (root / "src" / "pages" / "BulkCreate.js").write_text(
        "export default function BulkCreate() { return null; }\n", encoding="utf-8"
    )

    assert await _inspect(root) == ()


@pytest.mark.realrepo
@pytest.mark.asyncio
async def test_an_area_string_never_fires_because_nothing_is_matched_fuzzily(
    tmp_path: Path,
) -> None:
    """A plan that names areas rather than files says nothing this check can test.

    `expected_files_or_areas` holds both, and inferring which file inside `src/apiUtils` an
    area string meant is exactly the heuristic this check refuses to have. Not firing is the
    documented outcome, not a gap.
    """
    root = _checkout(tmp_path / "repo")
    (root / "src" / "apiUtils" / "bulkCreate.apiUtils.js").write_text(
        "export const BULK_CREATE_APPS = '/apps/bulk';\n", encoding="utf-8"
    )

    assert await _inspect(root, assigned_paths=("src", "src/apiUtils")) == ()


@pytest.mark.realrepo
@pytest.mark.asyncio
async def test_a_file_the_plan_expected_the_change_to_create_is_not_an_assigned_file(
    tmp_path: Path,
) -> None:
    """Condition 1 is membership in the checkout, and a plan often names what does not exist.

    A plan naming `src/apiUtils/bulkCreate.apiUtils.js` as the file to *write* would otherwise
    read as an assigned file the change failed to touch -- while the change was writing
    precisely it.
    """
    root = _checkout(tmp_path / "repo")
    (root / "src" / "apiUtils" / "bulkCreate.apiUtils.js").write_text(
        "export const BULK_CREATE_APPS = '/apps/bulk';\n", encoding="utf-8"
    )

    assert await _inspect(root, assigned_paths=("src/apiUtils/missing.apiUtils.js",)) == ()


@pytest.mark.realrepo
@pytest.mark.asyncio
async def test_an_added_file_of_another_kind_is_not_a_sibling_module(tmp_path: Path) -> None:
    """Condition 3 is a module of the same kind, which keeps a data file out of it.

    A generated document or a fixture written beside an assigned module is not the shape 183
    produced, and the suffix test is what says so without reading a byte of either file.
    """
    root = _checkout(tmp_path / "repo")
    (root / "src" / "apiUtils" / "endpoints.json").write_text(
        '{"bulkCreate": "/apps/bulk"}\n', encoding="utf-8"
    )

    assert await _inspect(root) == ()


@pytest.mark.realrepo
@pytest.mark.asyncio
async def test_a_deleted_assigned_file_is_not_an_untouched_one(tmp_path: Path) -> None:
    """`git status` names a deletion, and a deleted file is not a file left alone.

    A change that moved the code out of its assigned file has answered the question -- badly,
    perhaps, but review is what judges that. Reporting it here would tell a pass to declare
    the constant in a file the same change removed.
    """
    root = _checkout(tmp_path / "repo")
    (root / "src" / "apiUtils" / "allapps.apiUtils.js").unlink()
    (root / "src" / "apiUtils" / "bulkCreate.apiUtils.js").write_text(
        "export const BULK_CREATE_APPS = '/apps/bulk';\n", encoding="utf-8"
    )

    assert await _inspect(root) == ()


@pytest.mark.realrepo
@pytest.mark.asyncio
async def test_more_findings_than_the_bound_are_named_rather_than_dropped_in_silence(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A silent cap reads as "placement passed", which 46- calls out by name.

    Three assigned files each with a sibling; two findings are reported and the third is
    named in the log, so nobody can mistake a bounded run for a clean one.
    """
    root = _checkout(tmp_path / "repo")
    for index in range(3):
        (root / "src" / "apiUtils" / f"added{index}.apiUtils.js").write_text(
            f"export const ADDED_{index} = {index};\n", encoding="utf-8"
        )
    plan = (
        "src/apiUtils/allapps.apiUtils.js",
        "src/apiUtils/home.apiUtils.js",
        "src/pages/AllApps.js",
    )
    (root / "src" / "pages" / "Extra.js").write_text("export const extra = 1;\n", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="tools.assigned_file_conformance"):
        issues = await _inspect(root, assigned_paths=plan)

    assert len(issues) == 2, issues
    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert "assigned_file_findings_dropped" in logged
    assert "src/pages/AllApps.js" in logged


@pytest.mark.realrepo
@pytest.mark.asyncio
async def test_an_inspection_that_cannot_run_reports_nothing(tmp_path: Path) -> None:
    """A directory that is not a checkout blocks nothing: this is a hint, not a gate."""
    (tmp_path / "src").mkdir()

    issues = await RepositoryAssignedFileChecker(
        process_runner=AsyncioProcessRunner(),
        cancellation_token=MockCancellationToken(),
        timeout_seconds=60.0,
    ).issues(tmp_path, assigned_paths=_PLAN)

    assert issues == ()
