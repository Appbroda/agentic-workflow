"""The change's own tests inside the attempt, and the guard that lets them in at all.

The end-to-end proof of this lives in the `realrepo` tier, against real checkouts and a real
runner (`test_real_repository_gates.py`). These are the cheap propositions underneath it: the
default tier deselects `realrepo`, so without them a change to the marker lists below would
pass every test anybody runs by habit -- and those lists are the whole of the deterministic
half of the assertion guard.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from services.process_runner import ProcessResult
from tests.fixtures import node_javascript_repository
from tools.scoped_tests import (
    SCOPED_TEST_DIAGNOSTIC_PREFIX,
    NullScopedTestRunner,
    RepositoryScopedTestRunner,
    ScopedTestOutcome,
    assertion_shape,
    changed_test_paths,
    diagnostic_test_paths,
    is_scoped_test_diagnostic,
    weakened_test_paths,
)
from tools.technology_detection import inspect_repository_technology
from tools.validation_tools import (
    _WORKSPACE_REFERENCES_HEADER,
    MockValidationPlanBuilder,
    RepositoryValidationPlan,
    ValidationCommand,
    workspace_references_in_summary,
)

_SUITE = (
    "const { test } = require('node:test');\n"
    "const assert = require('node:assert');\n"
    "const { metricsRoute } = require('../src/routes/metrics');\n\n"
    "test('the metrics route reports a total', () => {\n"
    "  let body = null;\n"
    "  metricsRoute({}, { json: (value) => { body = value; } });\n"
    "  assert.deepStrictEqual(body, { count: 1, total: 3 });\n"
    "});\n"
)


class _ScriptedRunner:
    """Answer every command with one prepared result, and record what was asked."""

    def __init__(self, result: ProcessResult) -> None:
        """Hold the single answer this runner gives."""
        self._result = result
        self.commands: list[tuple[str, ...]] = []

    async def run(self, command: Any, cwd: Path, *_args: Any, **_kwargs: Any) -> ProcessResult:
        """Record the invocation and return the prepared result unchanged."""
        del cwd
        self.commands.append(tuple(command))
        return ProcessResult(
            command=tuple(command),
            return_code=self._result.return_code,
            stdout=self._result.stdout,
            stderr=self._result.stderr,
            duration_seconds=self._result.duration_seconds,
            timed_out=self._result.timed_out,
        )


def _plan(root: Path) -> RepositoryValidationPlan:
    """A repository declaring one narrowable test command and one lint command."""
    return RepositoryValidationPlan(
        repository_id="web",
        technology_profile=inspect_repository_technology(root, repository_id="web"),
        commands=[
            ValidationCommand(
                validation_type="lint", command=["npm", "run", "lint"], timeout_seconds=60
            ),
            ValidationCommand(
                validation_type="test",
                command=["npm", "run", "test"],
                timeout_seconds=60,
                accepts_test_paths=True,
                test_path_separator=["--"],
            ),
        ],
        source="repository_scripts",
    )


# --------------------------------------------------------------------------------------
# What the guard reads: a test file's assertion shape
# --------------------------------------------------------------------------------------


def test_repairing_scaffolding_leaves_the_assertion_shape_alone() -> None:
    """The C1 case. Correcting an unresolvable `require` changes nothing the guard reads.

    This is the line the whole part depends on: broken scaffolding is repairable inside the
    attempt precisely because repairing it does not touch what the suite claims.
    """
    broken = _SUITE.replace("../src/routes/metrics", "../src/routes/metric")

    assert assertion_shape(broken) == assertion_shape(_SUITE)
    assert weakened_test_paths({"test/a.test.js": broken}, {"test/a.test.js": _SUITE}) == []


def test_moving_what_a_test_expects_changes_its_shape() -> None:
    """The C2 case. Making the suite agree with the implementation is what must be caught."""
    weakened = _SUITE.replace("{ count: 1, total: 3 }", "{ count: 1 }")

    assert assertion_shape(weakened) != assertion_shape(_SUITE)
    assert weakened_test_paths({"test/a.test.js": _SUITE}, {"test/a.test.js": weakened}) == [
        "test/a.test.js"
    ]


def test_deleting_or_suppressing_a_test_is_weakening_too() -> None:
    """Neither shows up as an edited assertion, and both make a suite claim less.

    A guard that only compared assertion text would accept `test.skip` and an emptied file
    as clean, which is the obvious way around it and therefore the first thing to close.
    """
    skipped = _SUITE.replace("test('the metrics", "test.skip('the metrics")
    emptied = "const { test } = require('node:test');\n"

    assert weakened_test_paths({"a": _SUITE}, {"a": skipped}) == ["a"]
    assert weakened_test_paths({"a": _SUITE}, {"a": emptied}) == ["a"]
    # And a file the pass deleted outright, which arrives as an absent entry.
    assert weakened_test_paths({"a": _SUITE}, {}) == ["a"]


def test_moving_the_value_a_test_asserts_against_is_caught_too() -> None:
    """A claim is often written one line away from where it is made.

    `assert.deepStrictEqual(body, expected)` with `expected` weakened above it is the same
    weakening as editing the assertion in place, and a predicate that only read the line
    with `assert` on it would wave it through.
    """
    indirect = (
        "const assert = require('node:assert');\n"
        "const expected = { count: 1, total: 3 };\n"
        "test('the metrics route reports a total', () => {\n"
        "  assert.deepStrictEqual(route(), expected);\n"
        "});\n"
    )
    weakened = indirect.replace("{ count: 1, total: 3 }", "{ count: 1 }")

    assert weakened_test_paths({"a": indirect}, {"a": weakened}) == ["a"]


def test_reformatting_a_suite_is_not_weakening_it() -> None:
    """Indentation and ordering are not claims, so moving them must not stop the loop.

    Over-refusing costs an attempt and is the safe direction, but a guard that fired on any
    edit at all would refuse every legitimate scaffolding repair and Part C would be useless.
    """
    reformatted = "\n".join(f"  {line}" for line in _SUITE.splitlines()) + "\n"

    assert assertion_shape(reformatted) == assertion_shape(_SUITE)


def test_only_test_files_are_ever_handed_to_a_test_runner() -> None:
    """`node --test src/routes/status.js` runs a route as a suite, which answers nothing."""
    assert changed_test_paths(
        [
            "src/routes/metrics.js",
            "test/metrics.test.js",
            "package.json",
            "docs/metrics.md",
            "test/metrics.test.js",
        ]
    ) == ["test/metrics.test.js"]


# --------------------------------------------------------------------------------------
# What reaches the repair loop, and what must not
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_rejected_narrowed_run_becomes_one_tagged_diagnostic(tmp_path: Path) -> None:
    """A completed run that rejected the source is what the repair loop is allowed to see."""
    root = node_javascript_repository(tmp_path / "web")
    runner = _ScriptedRunner(
        ProcessResult(
            command=(),
            return_code=1,
            stdout="test/metrics.test.js:7\nAssertionError: expected total\n",
            stderr="",
            duration_seconds=0.2,
        )
    )

    outcome = await RepositoryScopedTestRunner(
        process_runner=runner, plan_builder=MockValidationPlanBuilder(_plan(root))
    ).run_changed_tests(root, ["src/routes/metrics.js", "test/metrics.test.js"])

    # Narrowed to the one suite, and the production file was never handed to the runner.
    assert runner.commands == [("npm", "run", "test", "--", "test/metrics.test.js")]
    assert outcome.commands == ("npm run test -- test/metrics.test.js",)
    assert len(outcome.diagnostics) == 1
    assert is_scoped_test_diagnostic(outcome.diagnostics[0])
    assert outcome.diagnostics[0].startswith(SCOPED_TEST_DIAGNOSTIC_PREFIX)
    assert "AssertionError: expected total" in outcome.diagnostics[0]


@pytest.mark.asyncio
async def test_a_run_that_could_not_complete_says_nothing_about_the_source(
    tmp_path: Path,
) -> None:
    """A timeout is not a verdict, and spending a repair pass on one is spending it on nothing.

    The command still ran, so it is still recorded -- but rewriting the file under test
    cannot make a suite fit inside its budget, and admitting this would put the repair loop
    to work on a question that was never answered.
    """
    root = node_javascript_repository(tmp_path / "web")
    runner = _ScriptedRunner(
        ProcessResult(
            command=(),
            return_code=None,
            stdout="",
            stderr="",
            duration_seconds=60.0,
            timed_out=True,
        )
    )

    outcome = await RepositoryScopedTestRunner(
        process_runner=runner, plan_builder=MockValidationPlanBuilder(_plan(root))
    ).run_changed_tests(root, ["test/metrics.test.js"])

    assert outcome.commands == ("npm run test -- test/metrics.test.js",)
    assert outcome.diagnostics == ()
    # And nothing structured either. A capacity failure admitted here would let a machine
    # problem accumulate on the unchanged-failure counter as though the source had not
    # moved, which is the AB-Feature-108 defect in a new place.
    assert outcome.validation_results == ()


@pytest.mark.asyncio
async def test_a_rejected_narrowed_run_records_the_run_it_actually_made(tmp_path: Path) -> None:
    """86- Part B. The gate's own verdict reaches the durable record, not just the prose.

    A gate-rejected attempt recorded `current_validation_results: []`, so the coarse failure
    counter -- which reads exactly that field, and correctly reads "empty" as *unknown* --
    broke its walk-back at every commit-gate rejection. AB-Feature-218's backend failed the
    same suite on four consecutive attempts and the counter never read more than two.

    Three properties, and each of them is load-bearing rather than decorative:
    `required` is true, because the counter reads only required failing commands and a run
    the gate *ended the attempt on* is as decisive as a command gets; the summary carries the
    reference section, because the files it names are the counted unit; and `passed` is
    false, because a passing entry contributes nothing.
    """
    root = _with_suite(node_javascript_repository(tmp_path / "web"))
    runner = _ScriptedRunner(
        ProcessResult(
            command=(),
            return_code=1,
            stdout="FAIL test/metrics.test.js\nAssertionError: expected total\n",
            stderr="",
            duration_seconds=0.2,
        )
    )

    outcome = await RepositoryScopedTestRunner(
        process_runner=runner, plan_builder=MockValidationPlanBuilder(_plan(root))
    ).run_changed_tests(root, ["test/metrics.test.js"])

    assert len(outcome.validation_results) == 1
    recorded = outcome.validation_results[0]
    assert recorded["command"] == "npm run test -- test/metrics.test.js"
    assert recorded["required"] is True
    assert recorded["passed"] is False
    assert recorded["validation_type"] == "test"
    assert recorded["result_code"]
    # The counted unit, read back by the same reader the counter uses over the same field.
    assert workspace_references_in_summary(recorded["stdout_summary"]) == ["test/metrics.test.js"]
    # One rejection, one diagnostic, one result: the prose half and the structured half are
    # the same verdict and are produced by the same predicate.
    assert len(outcome.diagnostics) == 1


@pytest.mark.asyncio
async def test_a_change_that_touched_no_test_file_runs_nothing(tmp_path: Path) -> None:
    """No suite changed, nothing to ask, and no subprocess started."""
    root = node_javascript_repository(tmp_path / "web")
    runner = _ScriptedRunner(
        ProcessResult(command=(), return_code=0, stdout="", stderr="", duration_seconds=0.1)
    )

    outcome = await RepositoryScopedTestRunner(
        process_runner=runner, plan_builder=MockValidationPlanBuilder(_plan(root))
    ).run_changed_tests(root, ["src/routes/metrics.js"])

    assert runner.commands == []
    assert outcome == await NullScopedTestRunner().run_changed_tests(root, ["src/routes/x.js"])


# --------------------------------------------------------------------------------------
# The suites the attempt was ASKED to fix, not only the ones it edited
# --------------------------------------------------------------------------------------


def _with_suite(root: Path) -> Path:
    """The suite the diagnostics name has to exist, because the reader re-checks that."""
    (root / "test" / "metrics.test.js").write_text(_SUITE, encoding="utf-8")
    return root


def _diagnostic_naming(*paths: str) -> str:
    """A blocking diagnostic carrying the reference section every failing summary appends."""
    return "\n".join(
        [
            f"{SCOPED_TEST_DIAGNOSTIC_PREFIX} (command=npm run test; code=1).",
            "  1 failing",
            "  BULK_APP_TRANSACTION_FAILED (500), expected BULK_APP_VALIDATION_FAILED (400)",
            "",
            _WORKSPACE_REFERENCES_HEADER,
            *paths,
        ]
    )


@pytest.mark.asyncio
async def test_a_production_only_remediation_runs_the_suite_it_was_asked_to_fix(
    tmp_path: Path,
) -> None:
    """AB-Feature-218's backend attempt 4, and attempt 5 after it.

    Both changed only `bulkApp.service.js` -- the right thing to change -- so
    `changed_test_paths` returned nothing, `scoped_test_commands` was recorded empty, the
    attempt passed this gate with the suite still red, and a full review cycle was spent
    rediscovering the identical failure. Checking the attempt that edits the test and
    skipping the attempt that fixes the production bug is a perverse incentive at exactly
    the point the model does the right thing.
    """
    root = _with_suite(node_javascript_repository(tmp_path / "web"))
    runner = _ScriptedRunner(
        ProcessResult(
            command=(),
            return_code=1,
            stdout="test/metrics.test.js:7\nAssertionError: expected 400, received 500\n",
            stderr="",
            duration_seconds=0.2,
        )
    )

    outcome = await RepositoryScopedTestRunner(
        process_runner=runner, plan_builder=MockValidationPlanBuilder(_plan(root))
    ).run_changed_tests(
        root,
        ["src/routes/metrics.js"],
        [_diagnostic_naming("test/metrics.test.js")],
    )

    assert runner.commands == [("npm", "run", "test", "--", "test/metrics.test.js")]
    assert outcome.commands == ("npm run test -- test/metrics.test.js",)
    # In-gate, which is strictly earlier and cheaper than the review cycle that fails it today.
    assert len(outcome.diagnostics) == 1
    assert is_scoped_test_diagnostic(outcome.diagnostics[0])


@pytest.mark.asyncio
async def test_a_diagnostic_naming_a_production_file_contributes_no_target(
    tmp_path: Path,
) -> None:
    """The union must not be able to hand a production file to a test runner.

    `node --test src/routes/metrics.js` runs a route as though it were a suite. The same
    `classify_file_change` predicate that narrows the attempt's own change set narrows this.
    """
    root = node_javascript_repository(tmp_path / "web")
    runner = _ScriptedRunner(
        ProcessResult(command=(), return_code=0, stdout="", stderr="", duration_seconds=0.1)
    )

    outcome = await RepositoryScopedTestRunner(
        process_runner=runner, plan_builder=MockValidationPlanBuilder(_plan(root))
    ).run_changed_tests(
        root, ["src/routes/metrics.js"], [_diagnostic_naming("src/routes/metrics.js")]
    )

    assert runner.commands == []
    assert outcome == ScopedTestOutcome()


@pytest.mark.asyncio
async def test_an_attempt_with_no_diagnostics_behaves_exactly_as_today(tmp_path: Path) -> None:
    """The floor stays: no diagnostics and no changed test file still runs nothing."""
    root = node_javascript_repository(tmp_path / "web")
    runner = _ScriptedRunner(
        ProcessResult(
            command=(),
            return_code=1,
            stdout="test/metrics.test.js:7\nAssertionError\n",
            stderr="",
            duration_seconds=0.2,
        )
    )
    gate = RepositoryScopedTestRunner(
        process_runner=runner, plan_builder=MockValidationPlanBuilder(_plan(root))
    )

    changed_only = await gate.run_changed_tests(root, ["test/metrics.test.js"])
    runner.commands.clear()
    nothing = await gate.run_changed_tests(root, ["src/routes/metrics.js"])

    assert changed_only.commands == ("npm run test -- test/metrics.test.js",)
    assert nothing == ScopedTestOutcome()
    assert runner.commands == []


@pytest.mark.asyncio
async def test_a_diagnostic_naming_a_deleted_test_file_contributes_no_target(
    tmp_path: Path,
) -> None:
    """The references were verified an attempt ago; the file may be gone by now.

    A renamed or deleted suite must contribute no target rather than a command that cannot
    run, and asking must not raise.
    """
    root = node_javascript_repository(tmp_path / "web")
    runner = _ScriptedRunner(
        ProcessResult(command=(), return_code=0, stdout="", stderr="", duration_seconds=0.1)
    )

    outcome = await RepositoryScopedTestRunner(
        process_runner=runner, plan_builder=MockValidationPlanBuilder(_plan(root))
    ).run_changed_tests(
        root,
        ["src/routes/metrics.js"],
        [_diagnostic_naming("test/deleted-in-a-previous-attempt.test.js", "../escape.test.js")],
    )

    assert runner.commands == []
    assert outcome == ScopedTestOutcome()


def test_the_two_halves_of_the_target_set_are_deduplicated(tmp_path: Path) -> None:
    """A suite the attempt both changed and was told to fix is one target, not two."""
    root = _with_suite(node_javascript_repository(tmp_path / "web"))

    named = diagnostic_test_paths([_diagnostic_naming("test/metrics.test.js")], root)

    assert named == ["test/metrics.test.js"]
    assert changed_test_paths(["test/metrics.test.js", "src/routes/metrics.js"]) == named
