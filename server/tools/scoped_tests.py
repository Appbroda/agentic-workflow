"""Run the change's own tests inside the attempt, and refuse a repair that weakens them.

Until now the first thing to run a repository's test suite was the Reviewer, which runs
*after* the Engineer has already declared itself done. So a failing test -- however trivial,
however visible in the file the attempt had just written -- was structurally guaranteed to
cost a full outer attempt and a 15-40 minute re-implementation.

The narrowed command `scoped_test_command` already builds is the cheap half of that verdict:
it asks the repository's own runner about the test files this attempt changed, and returns
`None` whenever narrowing would change the question. Running it inside the attempt lets the
existing bounded repair loop clear a broken fixture the way it clears a lint rule.

**The hazard this module exists to bound.** A failing test is ambiguous in a way a lint rule
is not: it may be broken scaffolding (mechanical) or wrong business logic (substantive). A
model told to make the diagnostics go away can satisfy a failing test by weakening its
assertion, and nothing downstream would ever catch that -- the suite would be green and
smaller. AB-Feature-182 is the precedent: three remediation attempts edited a test fixture
when the defect was in the product file.

So the admission comes with a guard in two halves, and the second half is deterministic:

* the repair instruction forbids editing assertions, expectations and test data, and says a
  correct-but-failing test is a substantive finding to hand back rather than to repair; and
* `weakened_test_paths` compares each test file's *assertion shape* across a repair pass. A
  pass that touched nothing but test files and changed that shape is rejected, the bytes it
  wrote are put back, and the loop stops with a named outcome.

Nothing here knows what any assertion means, and nothing here is repository-specific: the
command comes from `validation_plan()`, the narrowing from `scoped_test_command`, and the
markers below are framework conventions rather than any checkout's details.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol

from services.cancellation import CancellationToken, MockCancellationToken
from services.process_runner import ProcessRunner
from tools.channel_packages import mocked_channel_packages, specifier_names_package
from tools.file_tools import PathLike, WorkspacePathError, resolve_workspace_path
from tools.implementation_completeness import classify_file_change
from tools.reachability import import_bindings, module_specifiers
from tools.validation_tools import (
    ValidationPlanBuilder,
    ValidationResult,
    WorkspaceValidationTools,
    rejects_the_source,
    validation_summary,
    workspace_references_in_summary,
)

# How every diagnostic this module composes begins. The repair loop's eligibility predicate
# matches on this constant rather than on a copy of the sentence, so the producer of a
# diagnostic and the gate that admits it cannot drift apart -- which is the whole complaint
# against the two bare string literals the predicate started with.
SCOPED_TEST_DIAGNOSTIC_PREFIX = "The change's own tests failed"

# The named terminal outcome for a repair the assertion guard refused. It leaves the attempt
# as a sentence rather than as a raw test line, because every escape from an attempt has to
# be explainable from durable state alone.
ASSERTION_GUARD_OUTCOME = "SCOPED_TEST_ASSERTION_WEAKENED"

# Framework-neutral evidence that a line makes a claim about behaviour. Substring matching,
# lowercased, deliberately generous: `assert` alone covers Python's statement, Node's
# `assert.deepStrictEqual` and JUnit's `assertEquals`, and `expect` covers `expect(...)`
# together with the `const expected = ...` a suite asserts *against*, which is the ordinary
# way a claim is written one line away from where it is made.
#
# The asymmetry decides the generosity. Over-refusing costs one attempt and lands exactly
# where this repository stood before the change's own tests were run here at all;
# under-refusing leaves a quieter suite in the checkout permanently, and no later gate looks
# at it again. So a comment that merely says the word counts, and that is the right error.
#
# The limit, stated rather than left to be discovered: a claim whose data is indirected
# through a name none of these match -- `const golden = {...}` -- moves without being seen.
# That case is covered by the repair instruction and not by this predicate.
_ASSERTION_MARKERS = (
    "assert",
    "expect",
    ".should",
    "should(",
    "verify(",
    "must_equal",
)

# A test that is removed or switched off is weakened as surely as one whose assertion was
# edited, and neither shows up as a changed assertion line. Both halves are counted into the
# shape below: how many tests a file declares, and how many of them are suppressed.
_TEST_DECLARATION = re.compile(
    r"(?:^|[^\w.])(?:it|test|describe|context)\s*(?:\.\s*\w+\s*)?\(|(?:^|[^\w.])def\s+test\w*\s*\("
)
_SUPPRESSION_MARKERS = (
    ".skip(",
    ".todo(",
    ".only(",
    "xit(",
    "xtest(",
    "xdescribe(",
    "mark.skip",
    "mark.xfail",
    "@skip",
    "unittest.skip",
    "pytest.skip",
    "t.skip(",
)

# Framework-neutral evidence that a line asserts a *call* was made -- the command, not the
# effect. Lowercased substring matching, like `_ASSERTION_MARKERS` above and with the same
# stated generosity: `tohavebeencalled` covers every jest/vitest variant, `assert_called`
# covers unittest.mock's whole family, `calledwith`/`calledonce` cover sinon. A claim about a
# return value or a rendered result matches none of these, which is exactly the distinction
# the seam-mock detector needs.
_CALL_ASSERTION_MARKERS = (
    "tohavebeencalled",
    ".mock.calls",
    "assert_called",
    "assert_awaited",
    "assert_any_call",
    "assert_has_calls",
    "call_count",
    "call_args",
    "calledwith",
    "calledonce",
)
# How a mock's handle is bound on the mocking line itself: `with patch('x.y') as fake:`.
_MOCK_ALIAS = re.compile(r"\)\s*as\s+([A-Za-z_]\w*)")


@dataclass(frozen=True, slots=True)
class ScopedTestOutcome:
    """What the narrowed run did, and what it has to say about the source.

    ``commands`` is recorded even when nothing failed, because "the change's own tests were
    never run here" and "they ran and passed" are different facts about an attempt and a
    reader should not have to infer which one happened.

    ``validation_results`` is the structured half of the same verdict, in the one shape
    ``current_validation_results`` is persisted in, for the runs that completed and rejected
    the source. The diagnostics are what a repair pass reads; these are what the *coarse
    failure counter* reads, and until they existed a gate-rejected attempt recorded an empty
    validation list -- so 49-C's walk-back read it as "unknown", broke there, and
    AB-Feature-218's four-attempt run on one failing suite counted as two.
    """

    commands: tuple[str, ...] = ()
    diagnostics: tuple[str, ...] = ()
    validation_results: tuple[dict[str, Any], ...] = ()


class ScopedTestRunner(Protocol):
    """Ask the repository's own runner about the tests one attempt changed or was given."""

    async def run_changed_tests(
        self,
        workspace_root: PathLike,
        paths: Sequence[str],
        diagnostics: Sequence[str] = (),
    ) -> ScopedTestOutcome:
        """Return what ran and what it reported; never raise for an ordinary failure."""


class NullScopedTestRunner:
    """Run nothing, which is what a composition with no configured runner must do."""

    async def run_changed_tests(
        self,
        workspace_root: PathLike,
        paths: Sequence[str],
        diagnostics: Sequence[str] = (),
    ) -> ScopedTestOutcome:
        """Report that no narrowed test command was available."""
        del workspace_root, paths, diagnostics
        return ScopedTestOutcome()


class RepositoryScopedTestRunner:
    """Run the checkout's own test command, narrowed to the tests this attempt changed.

    Deliberately unjournaled. `WorkspaceValidationTools` keys a journaled validation on the
    working-tree revision and the command, and the Reviewer runs the very same narrowed
    command a few minutes later -- so an attempt that needed no repair would produce an
    identical key and hand the Reviewer this run's verdict instead of its own. Fast feedback
    inside the attempt must never become the authority outside it.
    """

    def __init__(
        self,
        *,
        process_runner: ProcessRunner | None = None,
        cancellation_token: CancellationToken | None = None,
        plan_builder: ValidationPlanBuilder | None = None,
        timeout_seconds: float | None = None,
        repository_id: str | None = None,
    ) -> None:
        """Bind the execution boundaries; nothing here reaches beyond the workspace."""
        self._process_runner = process_runner
        self._cancellation_token = cancellation_token or MockCancellationToken()
        self._plan_builder = plan_builder
        self._timeout = timeout_seconds
        self._repository_id = repository_id

    async def run_changed_tests(
        self,
        workspace_root: PathLike,
        paths: Sequence[str],
        diagnostics: Sequence[str] = (),
    ) -> ScopedTestOutcome:
        """Run the narrowed command for each configured test command, and report failures.

        Only a command that *completed and rejected the source* becomes a diagnostic. A run
        that timed out or exhausted the machine says nothing about the change -- rewriting
        the file under test cannot make a suite fit in memory -- and admitting one to the
        repair loop would spend a coding budget on a verdict that was never available.

        **The suites this attempt was ASKED to fix count, not only the ones it edited.**
        Narrowing to the change's own test files alone checks an attempt that rewrote the
        test and skips an attempt that fixed the production bug -- a perverse incentive at
        exactly the point the model does the right thing. AB-Feature-218's backend attempts
        4 and 5 each changed only `bulkApp.service.js`, so `changed_test_paths` returned
        nothing, `scoped_test_commands` was recorded empty, both passed this gate with the
        suite still red, and each then spent a full review cycle rediscovering the identical
        failure. ``diagnostics`` are this attempt's blocking diagnostics, passed in by the
        caller so this runner stays a function of its arguments.

        The floor stays. An attempt with no diagnostics and no changed test file still runs
        nothing, which is correct and is what it does today.
        """
        targets = list(
            dict.fromkeys(
                [*changed_test_paths(paths), *diagnostic_test_paths(diagnostics, workspace_root)]
            )
        )
        if not targets:
            return ScopedTestOutcome()
        tools = WorkspaceValidationTools(
            workspace_root,
            repository_id=self._repository_id,
            plan_builder=self._plan_builder,
            cancellation_token=self._cancellation_token,
            process_runner=self._process_runner,
            # No operation executor: see the class docstring.
            operation_executor=None,
        )
        results = await tools.run_changed_tests_async(targets, timeout_seconds=self._timeout)
        commands = tuple(" ".join(result.command) for result in results)
        rejecting = tuple(result for result in results if rejects_the_source(result))
        diagnostics = tuple(scoped_test_diagnostic(result) for result in rejecting)
        return ScopedTestOutcome(
            commands=commands,
            diagnostics=diagnostics,
            # The structured half of the same verdict, from the same predicate. Promoted to
            # required for the reason `run_validations_async` promotes its own narrowed run:
            # this is the repository's own runner rejecting the source, however few files it
            # was given -- and here it is what *ended the attempt*, which is as decisive as a
            # required command gets. The coarse failure counter reads only required failing
            # commands, so recording these as optional would record them into a field nothing
            # reads.
            validation_results=tuple(
                validation_summary(replace(result, required=True)) for result in rejecting
            ),
        )


def scoped_test_diagnostic(result: ValidationResult) -> str:
    """Compose one repair-loop diagnostic from a narrowed run that rejected the source.

    The bounded summaries rather than the raw streams: they are already redacted, already
    length-bounded, and already carry the `[workspace files named by this output]` section
    that lets the repair prompt quote the source around the failing location.
    """
    headline = (
        f"{SCOPED_TEST_DIAGNOSTIC_PREFIX} (command={' '.join(result.command)}; "
        f"code={result.result_code})."
    )
    excerpt = "\n".join(part for part in (result.stdout_summary, result.stderr_summary) if part)
    return f"{headline}\n{excerpt}" if excerpt.strip() else headline


def is_scoped_test_diagnostic(diagnostic: str) -> bool:
    """Say whether one diagnostic is this module's, and therefore about a failing test."""
    return diagnostic.startswith(SCOPED_TEST_DIAGNOSTIC_PREFIX)


def changed_test_paths(paths: Sequence[str]) -> list[str]:
    """Keep only the test files out of an attempt's change set, in first-seen order.

    Handing a production file to a test runner is not narrowing, it is asking a different
    question: `node --test src/routes/status.js` runs a route as though it were a suite.
    `classify_file_change` is the same convention-based split the completeness evidence and
    the Reviewer's own narrowing already use.
    """
    return list(
        dict.fromkeys(path for path in paths if classify_file_change(path) == "test").keys()
    )


def diagnostic_test_paths(diagnostics: Sequence[str], workspace_root: PathLike) -> list[str]:
    """Keep the test files this attempt's blocking diagnostics named, in first-seen order.

    The other half of "which suites is this attempt about". `changed_test_paths` reads what
    the attempt WROTE; this reads what it was TOLD, which is the only signal available when
    a remediation correctly changes production code and nothing else.

    Read through `workspace_references_in_summary`, the same reader `unchanged_failure`
    already uses over the same durable summaries, so a diagnostic's file set has one
    interpretation in this platform rather than two. Filtered by the same
    `classify_file_change` predicate `changed_test_paths` applies, because the union must not
    be able to hand a production file to a test runner: `node --test src/routes/status.js`
    runs a route as though it were a suite.

    Existence is re-checked here, against this workspace, now. The references were verified
    when the summary was written -- an attempt ago -- and a file the attempt has since
    deleted or renamed must contribute no target rather than a command that cannot run.
    """
    root = resolve_workspace_path(workspace_root, ".")
    named: list[str] = []
    for diagnostic in diagnostics:
        for path in workspace_references_in_summary(diagnostic):
            if path in named or classify_file_change(path) != "test":
                continue
            try:
                resolved = resolve_workspace_path(root, path)
            except (WorkspacePathError, ValueError, OSError):
                continue
            if resolved.is_file():
                named.append(path)
    return named


def assertion_shape(source: str) -> tuple[str, ...]:
    """Describe what a test file claims, ignoring everything about how it sets up to claim it.

    Two files with the same shape make the same assertions, declare the same number of tests
    and suppress the same number of them. Repairing broken scaffolding -- an import that does
    not resolve, a mock factory that throws, a setup step that never ran -- leaves the shape
    alone, and that is exactly the repair this loop is allowed to make.

    Assertion lines are sorted and whitespace-normalized, so reformatting or reordering is
    not read as weakening. Nothing here parses the language or knows what any claim means.
    """
    declarations = 0
    suppressions = 0
    assertions: list[str] = []
    for line in source.splitlines():
        lowered = line.lower()
        declarations += len(_TEST_DECLARATION.findall(line))
        suppressions += sum(lowered.count(marker) for marker in _SUPPRESSION_MARKERS)
        if any(marker in lowered for marker in _ASSERTION_MARKERS):
            assertions.append(" ".join(line.split()))
    return (
        f"declarations={declarations}",
        f"suppressions={suppressions}",
        *sorted(assertions),
    )


def weakened_test_paths(before: Mapping[str, str], after: Mapping[str, str]) -> list[str]:
    """Return the test files whose assertion shape a repair pass changed.

    Only files present on both sides are compared. A repair that adds a new test file cannot
    have weakened an assertion that was never there, and one that deletes a file it never had
    is not something this comparison can observe -- the deletion case arrives here as a file
    present in ``before`` and absent from ``after``, whose shape is empty and therefore differs.
    """
    changed: list[str] = []
    for path, original in before.items():
        if assertion_shape(original) != assertion_shape(after.get(path, "")):
            changed.append(path)
    return sorted(changed)


def assertion_guard_diagnostic(paths: Sequence[str]) -> str:
    """Say, as a sentence a person can act on, why a repair pass was thrown away."""
    return (
        f"{ASSERTION_GUARD_OUTCOME}: the in-attempt repair changed nothing but test files, "
        f"and it changed what {', '.join(paths)} asserts. A failing test is either broken "
        "scaffolding, which this loop may repair, or a correct test rejecting the "
        "implementation, which it may not silently rewrite. The repair was discarded and "
        "those files were restored to the bytes the attempt wrote. Whether the test or the "
        "implementation is wrong is a judgement for review, not for a mechanical repair."
    )


def seam_mock_note(package: str, path: str) -> str:
    """Say, in one stable sentence, that this file verifies a seam only against its mock."""
    return f"seam '{package}' is verified only against its own mock in {path}"


def seam_mock_notes(path: str, source: str) -> tuple[str, ...]:
    """Name each channel package this changed test both mocks and command-asserts on.

    A test that replaces a channel package with a double and then asserts on that double --
    `toHaveBeenCalled*` on it, `assert_called*` on it -- is command-asserting at the seam: it
    can only ever prove the code called the mock, never that a request would leave the
    building. AB-Feature-206's unit test did exactly this, and to the reviewer its green run
    read as spec conformance for the one behaviour it structurally cannot verify.

    Sibling of `assertion_shape`, and like it, deliberately textual: nothing here parses the
    language or knows what any claim means. The output is a named review-evidence note and
    nothing else. Under 47-D's law the in-attempt layer informs and never fails the attempt:
    the note does not block the test from running and does not forbid seam mocks -- unit
    tests may mock channels freely -- it only refuses to let a mocked seam impersonate seam
    verification when a reviewer weighs it.

    The tie between the double and the assertion is by name where a name is derivable -- the
    bindings of the import that names the package, an `as` alias on the mocking line, the
    package's own spelling -- and by file where none is: a Python mock injected as a decorator
    argument binds a parameter this scan cannot see, and missing that note is the expensive
    direction, so a file that mocks a channel and command-asserts with no derivable name is
    noted rather than waved through.
    """
    mocked = mocked_channel_packages(path, source)
    if not mocked:
        return ()
    assertion_lines = [
        line
        for line in source.splitlines()
        if any(marker in line.lower() for marker in _CALL_ASSERTION_MARKERS)
    ]
    if not assertion_lines:
        return ()
    notes: list[str] = []
    for package in mocked:
        names = _seam_double_names(path, source, package)
        if not names or any(
            re.search(rf"\b{re.escape(name)}\b", line)
            for name in sorted(names)
            for line in assertion_lines
        ):
            notes.append(seam_mock_note(package, path))
    return tuple(notes)


def _seam_double_names(path: str, source: str, package: str) -> set[str]:
    """Return the local names this file could call the mocked package by.

    Empty means no name was derivable at all -- a mock reached purely through a string
    target, its handle injected somewhere this scan cannot see -- and the caller treats that
    as "cannot rule the assertion out" rather than "ruled out". A file that *imports* the
    package binds real names, and those are required to appear in the assertion; one known
    miss is stated rather than hidden: a decorator-injected handle in a file that also
    imports the package is tied to the import's names, not the handle, and can be missed.
    """
    names: set[str] = set()
    for line in source.splitlines():
        if any(
            specifier_names_package(specifier, package, separator)
            for specifier in module_specifiers(line)
            for separator in ("/", ".")
        ):
            names |= import_bindings(line)
            if re.fullmatch(r"[A-Za-z_]\w*", package):
                # `axios.post` under jest's auto-mock: the imported package's own spelling
                # is a usable handle wherever it is a bare identifier.
                names.add(package)
        if package in mocked_channel_packages(path, line):
            alias = _MOCK_ALIAS.search(line)
            if alias is not None:
                names.add(alias.group(1))
    return names


__all__ = [
    "ASSERTION_GUARD_OUTCOME",
    "SCOPED_TEST_DIAGNOSTIC_PREFIX",
    "NullScopedTestRunner",
    "RepositoryScopedTestRunner",
    "ScopedTestOutcome",
    "ScopedTestRunner",
    "assertion_guard_diagnostic",
    "assertion_shape",
    "changed_test_paths",
    "is_scoped_test_diagnostic",
    "scoped_test_diagnostic",
    "seam_mock_note",
    "seam_mock_notes",
    "weakened_test_paths",
]
