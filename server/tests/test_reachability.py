"""The reachability inspection both layers share, and the three blind spots it closed.

`_unreachable_addition_issues` used to be a method of the runtime and nothing else could ask
it anything. It is now `tools.reachability`, called from two places: the Engineer, inside its
own attempt, as feedback its repair loop can act on; and the runtime, unchanged, as the
authoritative gate. The end-to-end proof of the two layers is in the `realrepo` tier
(`test_real_repository_gates.py`, D1 and D4); this file is the inspection itself.

The propositions here divide in two, deliberately:

* the export parser and the diagnostic tag are pure, so they run in the **default** tier.
  Without that, a change to the patterns or the generic-name list -- which are the whole of
  how a symbol is now resolved -- would pass every test anybody runs by habit.
* D2 and D3 drive real `git status`, `git grep`, `git show` and `git ls-files` over real
  checkouts, so they are marked `realrepo` with the rest of this spec's acceptance list.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import pytest

from services.cancellation import MockCancellationToken
from services.process_runner import AsyncioProcessRunner
from tools.reachability import (
    REACHABILITY_DIAGNOSTIC_PREFIX,
    NullReachabilityChecker,
    ReachabilityOutcome,
    RepositoryReachabilityChecker,
    candidate_symbols,
    exported_names,
    is_reachability_diagnostic,
    reachability_diagnostic,
    unreachable_addition_issues,
)

# --------------------------------------------------------------------------------------
# Blind spot 3: the symbol is the file's exports, not its filename
# --------------------------------------------------------------------------------------


def test_the_public_surface_is_read_from_the_forms_repositories_actually_write() -> None:
    """Every export shape below appeared in a checkout this platform has run against."""
    source = (
        "export function formatCurrency(value) {}\n"
        "export default class LedgerView {}\n"
        "export const TAX_BANDS = [];\n"
        "exports.renderInvoice = () => {};\n"
        "export { settleInvoice, voidInvoice as cancelInvoice };\n"
    )

    assert exported_names(source) == (
        "formatCurrency",
        "LedgerView",
        "TAX_BANDS",
        "renderInvoice",
        "settleInvoice",
        "voidInvoice",
        "cancelInvoice",
    )


def test_a_python_module_declares_its_surface_at_column_zero() -> None:
    """A method is not a module export, and the indentation is the only thing that says so."""
    source = (
        "class StatusFeed:\n"
        "    def refresh(self):\n"
        "        return None\n"
        "\n"
        "def build_status_feed():\n"
        "    return StatusFeed()\n"
    )

    assert exported_names(source) == ("build_status_feed", "StatusFeed")


def test_a_string_literal_in_an_export_list_contributes_no_names() -> None:
    """`module.exports = { greet: () => 'hello world' }` exports one name, not three.

    Every extra name makes this check *quieter* about a genuinely dead module, because a
    reference to any candidate counts as reaching it. So a name invented out of a string is
    not a harmless imprecision: it is a way for an unreachable module to look wired.
    """
    assert exported_names("module.exports = { greet: () => 'hello world' };\n") == ("greet",)


def test_the_filename_stem_is_kept_as_well_as_the_exports(tmp_path: Path) -> None:
    """Both are needed, because a reference is written either as a symbol or as a path.

    `import { formatCurrency } from './utils/helpers'` names the export; `require('./utils/
    helpers')` names only the stem. Resolving one and not the other is how -072 and -073 both
    shipped a page nothing rendered: the stem was checked, `getServerHealthHistory` contained
    it, and this check stayed silent.
    """
    module = tmp_path / "src" / "utils" / "helpers.js"
    module.parent.mkdir(parents=True)
    module.write_text("export const formatCurrency = () => {};\n", encoding="utf-8")

    assert candidate_symbols(tmp_path, "src/utils/helpers.js") == ("formatCurrency", "helpers")


def test_a_generic_export_name_is_not_evidence_and_is_dropped(tmp_path: Path) -> None:
    """`module.exports = router` must not make every router in the checkout a referrer.

    This is the failure the stem-only version had, in a new place: a word that matches
    everywhere reports a dead module as reached. The stem survives, so the file is still
    checked -- just not against a name that means nothing.
    """
    module = tmp_path / "src" / "routes" / "invoices.js"
    module.parent.mkdir(parents=True)
    module.write_text("const router = express.Router();\nmodule.exports = router;\n", "utf-8")

    assert candidate_symbols(tmp_path, "src/routes/invoices.js") == ("invoices",)


# --------------------------------------------------------------------------------------
# What the Engineer's repair loop is allowed to see
# --------------------------------------------------------------------------------------


def test_the_tag_leaves_the_runtimes_own_sentence_byte_identical() -> None:
    """The two layers report the same words, so a repair and a person read the same finding.

    Only the in-attempt copy is tagged, and only so the repair loop's eligibility predicate
    can recognise it. The runtime's blocking issues are read by people and classified by the
    retry policy, and this move must not change one of their bytes.
    """
    issue = "src/routes/metrics.js was added but no production file refers to it."

    tagged = reachability_diagnostic(issue)

    assert is_reachability_diagnostic(tagged)
    assert tagged.startswith(REACHABILITY_DIAGNOSTIC_PREFIX)
    assert issue in tagged
    assert not is_reachability_diagnostic(issue)


@pytest.mark.asyncio
async def test_a_composition_with_no_checker_inspects_nothing(tmp_path: Path) -> None:
    """A deployment or a double that wires no checker behaves exactly as before this part."""
    outcome = await NullReachabilityChecker().issues(tmp_path, assigned_paths=("src",))

    assert outcome.issues == ()
    assert outcome.repairs == ()
    assert outcome.dropped_candidates == ()


# --------------------------------------------------------------------------------------
# D2 and D3, against real checkouts
# --------------------------------------------------------------------------------------


def _git(root: Path, *arguments: str) -> None:
    """Run one git command for real, because every answer here comes out of a checkout."""
    subprocess.run(("git", *arguments), cwd=root, check=True, capture_output=True, timeout=30)


def _wired_checkout(root: Path) -> Path:
    """A checkout whose route directory is wired by hand, which is what makes it answerable.

    `src/index.js` reaches `src/routes/status.js` by path, so this directory is not scanned by
    convention and an unreferenced newcomer in it is genuinely dead code. Committed, because
    the inspection compares the working tree against HEAD.
    """
    (root / "src" / "routes").mkdir(parents=True)
    (root / "src" / "index.js").write_text(
        "const { statusRoute } = require('./routes/status');\nmodule.exports = { statusRoute };\n",
        encoding="utf-8",
    )
    (root / "src" / "routes" / "status.js").write_text(
        "function statusRoute(request, response) {\n  response.json({ status: 'ok' });\n}\n\n"
        "module.exports = { statusRoute };\n",
        encoding="utf-8",
    )
    _git(root, "init", "--quiet")
    _git(root, "config", "user.email", "engineer@example.test")
    _git(root, "config", "user.name", "Engineer")
    _git(root, "add", "-A")
    _git(root, "commit", "--quiet", "-m", "A hand-wired route directory")
    return root


async def _inspect(root: Path, *, assigned_paths: tuple[str, ...] = ()) -> ReachabilityOutcome:
    """Ask the shared inspection about this checkout, exactly as both layers ask it."""
    return await unreachable_addition_issues(
        root,
        runner=AsyncioProcessRunner(),
        timeout=60.0,
        cancellation_token=MockCancellationToken(),
        assigned_paths=assigned_paths,
    )


@pytest.mark.realrepo
@pytest.mark.asyncio
async def test_an_added_export_in_a_modified_file_is_reported(tmp_path: Path) -> None:
    """D2. The new export lives in an existing file and nothing calls it.

    This was invisible. The whole inspection filtered `git status` to `??`/`A ` and reasoned
    about *files*, so a helper exported for a caller that was never written read as "wiring
    passed" -- the same unfinished work as a component nobody renders, in the one shape the
    gate could not see.

    Note what makes this answerable and not a guess: `src/routes/status.js` mentions
    `formatStatus` exactly twice, on the line declaring it and on the line exporting it, and
    neither is a call. A check that counted occurrences would accept every added export.
    """
    root = _wired_checkout(tmp_path / "repo")
    (root / "src" / "routes" / "status.js").write_text(
        "function statusRoute(request, response) {\n  response.json({ status: 'ok' });\n}\n\n"
        "function formatStatus(status) {\n  return { status };\n}\n\n"
        "module.exports = { statusRoute, formatStatus };\n",
        encoding="utf-8",
    )

    outcome = await _inspect(root)

    assert len(outcome.issues) == 1, outcome.issues
    assert "src/routes/status.js now exports formatStatus" in outcome.issues[0]
    assert outcome.repairs[0]["problem"] == "exported_but_unreferenced"
    assert outcome.repairs[0]["added_path"] == "src/routes/status.js"
    assert outcome.repairs[0]["symbol"] == "formatStatus"
    # No host is named, because the plan named none and this shape has no structural answer
    # worth stating. A guess force-included as the answer is the AB-Feature-112 failure.
    assert "target_path" not in outcome.repairs[0]


@pytest.mark.realrepo
@pytest.mark.asyncio
async def test_an_added_export_something_calls_is_not_reported(tmp_path: Path) -> None:
    """The other half of D2, and the one that decides whether it can ship.

    An addition reported when it is genuinely wired costs a whole attempt to disprove, so
    over-reporting here is not the safe direction it is elsewhere. Once a production file
    calls the new name, the finding has to go away.
    """
    root = _wired_checkout(tmp_path / "repo")
    (root / "src" / "routes" / "status.js").write_text(
        "function statusRoute(request, response) {\n  response.json({ status: 'ok' });\n}\n\n"
        "function formatStatus(status) {\n  return { status };\n}\n\n"
        "module.exports = { statusRoute, formatStatus };\n",
        encoding="utf-8",
    )
    (root / "src" / "index.js").write_text(
        "const { statusRoute, formatStatus } = require('./routes/status');\n"
        "module.exports = { statusRoute, banner: () => formatStatus('ok') };\n",
        encoding="utf-8",
    )

    outcome = await _inspect(root)

    assert outcome.issues == ()


@pytest.mark.realrepo
@pytest.mark.asyncio
async def test_a_new_export_in_a_file_nothing_imports_says_nothing(tmp_path: Path) -> None:
    """A package entry point's surface is reached from outside the checkout, so this is quiet.

    `src/index.js` is what consumes this repository imports, and no amount of grepping inside
    it can establish whether its exports are called. The same silence covers a
    convention-loaded module -- a Flask blueprint, a framework page -- which runs without
    anything importing it. Both are correct code, and reporting either would be a false
    positive that costs a real attempt.
    """
    root = _wired_checkout(tmp_path / "repo")
    (root / "src" / "index.js").write_text(
        "const { statusRoute } = require('./routes/status');\n"
        "function describeService() {\n  return 'status';\n}\n"
        "module.exports = { statusRoute, describeService };\n",
        encoding="utf-8",
    )

    outcome = await _inspect(root)

    assert outcome.issues == ()


@pytest.mark.realrepo
@pytest.mark.asyncio
async def test_a_decorated_handler_is_wired_by_its_decorator(tmp_path: Path) -> None:
    """`@app.route` above a declaration *is* the wiring, in every framework that offers one.

    The only place this inspection knows a convention exists, and it only ever makes it
    quieter. Without this, every route a Python service adds to an existing module would be
    reported as unreachable.
    """
    root = tmp_path / "repo"
    (root / "app").mkdir(parents=True)
    (root / "app" / "__init__.py").write_text('"""Service."""\n', encoding="utf-8")
    (root / "app" / "routes.py").write_text(
        "from app.server import app\n\n\n"
        "@app.route('/status')\n"
        "def read_status():\n"
        '    return {"status": "ok"}\n',
        encoding="utf-8",
    )
    (root / "app" / "server.py").write_text(
        '"""Entry point."""\n\nimport app.routes\n\napp = object()\n', encoding="utf-8"
    )
    _git(root, "init", "--quiet")
    _git(root, "config", "user.email", "engineer@example.test")
    _git(root, "config", "user.name", "Engineer")
    _git(root, "add", "-A")
    _git(root, "commit", "--quiet", "-m", "A decorated route module")
    (root / "app" / "routes.py").write_text(
        "from app.server import app\n\n\n"
        "@app.route('/status')\n"
        "def read_status():\n"
        '    return {"status": "ok"}\n\n\n'
        "@app.route('/metrics')\n"
        "def read_metrics():\n"
        '    return {"count": 1}\n',
        encoding="utf-8",
    )

    outcome = await _inspect(root)

    assert outcome.issues == ()


@pytest.mark.realrepo
@pytest.mark.asyncio
async def test_the_candidate_cap_names_everything_it_never_asked_about(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """D3. More candidates than the bound: the drop is loud, and the run does not read as passed.

    A hard cap of eight applied as `added[:8]` left the rest unchecked and logged nothing, so
    a change adding more than eight new production modules read as "wiring passed". 46- calls
    a silent cap out by name and this was exactly that kind.

    Two channels, because a log line alone is not an answer anybody can look up later: the
    dropped paths are named in the log *and* returned, and the Engineer records the returned
    list on the attempt's own artifact.
    """
    root = _wired_checkout(tmp_path / "repo")
    for index in range(11):
        (root / "src" / "routes" / f"widget{index:02d}.js").write_text(
            f"function widget{index:02d}() {{\n  return {index};\n}}\n\n"
            f"module.exports = {{ widget{index:02d} }};\n",
            encoding="utf-8",
        )

    with caplog.at_level(logging.WARNING, logger="tools.reachability"):
        outcome = await _inspect(root)

    assert len(outcome.examined_candidates) == 8
    assert len(outcome.dropped_candidates) == 3
    # Named, not counted: "three candidates were dropped" cannot be acted on.
    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert "reachability_candidates_dropped" in logged
    for path in outcome.dropped_candidates:
        assert path in logged
    # And the truncated run is nothing like a clean one: it still reports what it did examine.
    assert len(outcome.issues) == 8


@pytest.mark.realrepo
@pytest.mark.asyncio
async def test_an_added_module_is_resolved_by_its_export_and_not_only_its_stem(
    tmp_path: Path,
) -> None:
    """A file whose exports do not share its filename is now checked against both.

    The importer here names the export and never the stem -- `require('./routes')` resolves
    through a directory -- so a stem-only check would call a correctly wired module dead and
    send the attempt back to wire something that is already wired.
    """
    root = _wired_checkout(tmp_path / "repo")
    (root / "src" / "routes" / "collections.js").write_text(
        "export function assembleCollection() {\n  return [];\n}\n", encoding="utf-8"
    )
    (root / "src" / "index.js").write_text(
        "const { statusRoute } = require('./routes/status');\n"
        "const { assembleCollection } = require('./routes');\n"
        "module.exports = { statusRoute, listing: () => assembleCollection() };\n",
        encoding="utf-8",
    )

    outcome = await _inspect(root)

    assert outcome.issues == ()


@pytest.mark.realrepo
@pytest.mark.asyncio
async def test_a_substring_of_a_longer_identifier_is_still_not_a_reference(
    tmp_path: Path,
) -> None:
    """`-w` and case sensitivity survive the move, and the -072/-073 bug stays fixed.

    Resolving more names could only have been an improvement if it had not relaxed the match.
    `getStatusFormatterHistory` contains the stem of `statusformatter.js` and reaches nothing;
    three consoles shipped unreachable pages while this check stayed silent about exactly that.
    """
    root = _wired_checkout(tmp_path / "repo")
    (root / "src" / "routes" / "statusformatter.js").write_text(
        "function shape(status) {\n  return { status };\n}\n\nmodule.exports = { shape };\n",
        encoding="utf-8",
    )
    (root / "src" / "helpers.js").write_text(
        "const getStatusFormatterHistory = () => [];\n"
        "module.exports = { getStatusFormatterHistory };\n",
        encoding="utf-8",
    )

    outcome = await _inspect(root)

    assert any("src/routes/statusformatter.js" in issue for issue in outcome.issues), outcome.issues


@pytest.mark.realrepo
@pytest.mark.asyncio
async def test_a_wiring_satisfied_only_by_text_is_recorded_and_still_passes(
    tmp_path: Path,
) -> None:
    """216's accepted wiring, both textual shapes: a declared-only const, and a quoted path.

    `const gather_metrics = 'src/utils/gather_metrics.js::gather_metrics'` names the symbol
    outside any string -- the const is named after it -- so quote-stripping alone sees nothing;
    what establishes "textual" is that the name is declared and exported and never run. The
    gate still passes: the fact is recorded, because whether a route-table string is real
    wiring is a repository convention no unattended gate may rule on.
    """
    root = _wired_checkout(tmp_path / "repo")
    (root / "src" / "utils").mkdir()
    (root / "src" / "utils" / "gather_metrics.js").write_text(
        "function gather_metrics() {\n  return [];\n}\n\nmodule.exports = { gather_metrics };\n",
        encoding="utf-8",
    )
    (root / "src" / "index.js").write_text(
        "const { statusRoute } = require('./routes/status');\n"
        "const gather_metrics = 'src/utils/gather_metrics.js::gather_metrics';\n"
        "module.exports = { statusRoute, gather_metrics };\n",
        encoding="utf-8",
    )

    outcome = await _inspect(root)

    assert not any("gather_metrics" in issue for issue in outcome.issues), outcome.issues
    assert [item["wiring_reference_kind"] for item in outcome.textual_wirings] == ["textual_only"]
    assert outcome.textual_wirings[0]["added_path"] == "src/utils/gather_metrics.js"
    assert outcome.textual_wirings[0]["referrer"] == "src/index.js"


@pytest.mark.realrepo
@pytest.mark.asyncio
async def test_a_referrer_that_imports_and_runs_the_symbol_is_not_marked_textual(
    tmp_path: Path,
) -> None:
    """Real wiring records nothing: the marker exists only for the text-only shape."""
    root = _wired_checkout(tmp_path / "repo")
    (root / "src" / "utils").mkdir()
    (root / "src" / "utils" / "gather_metrics.js").write_text(
        "function gather_metrics() {\n  return [];\n}\n\nmodule.exports = { gather_metrics };\n",
        encoding="utf-8",
    )
    (root / "src" / "index.js").write_text(
        "const { statusRoute } = require('./routes/status');\n"
        "const { gather_metrics } = require('./utils/gather_metrics');\n"
        "module.exports = { statusRoute, metrics: () => gather_metrics() };\n",
        encoding="utf-8",
    )

    outcome = await _inspect(root)

    assert outcome.issues == ()
    assert outcome.textual_wirings == ()


@pytest.mark.realrepo
@pytest.mark.asyncio
async def test_an_added_conftest_is_not_demanded_wired_into_production(tmp_path: Path) -> None:
    """Test infrastructure that nothing imports is correct, not dead (83-, AB-Feature-216).

    216's coder added `server/conftest.py`; it classified as production, this gate demanded a
    production referrer and named the OAuth model as the strongest candidate, and attempt 3
    obeyed with a string literal. A runner-wired file is reached by its runner's convention,
    which no import can show.
    """
    root = _wired_checkout(tmp_path / "repo")
    (root / "conftest.py").write_text(
        "def pytest_collect_file(parent, path):\n    return None\n", encoding="utf-8"
    )

    outcome = await _inspect(root)

    assert not any("conftest.py" in issue for issue in outcome.issues), outcome.issues


@pytest.mark.realrepo
@pytest.mark.asyncio
async def test_an_inspection_that_cannot_run_reports_nothing(tmp_path: Path) -> None:
    """A directory that is not a checkout blocks nothing: this is fast feedback, not a gate.

    The authoritative answer is the runtime's, and failing an attempt because the inspection
    itself could not run would replace a wiring defect with a platform one.
    """
    (tmp_path / "src").mkdir()

    outcome = await RepositoryReachabilityChecker(
        process_runner=AsyncioProcessRunner(),
        cancellation_token=MockCancellationToken(),
        timeout_seconds=60.0,
    ).issues(tmp_path)

    assert outcome.issues == ()
