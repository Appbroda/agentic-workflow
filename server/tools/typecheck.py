"""Run the checkout's own typecheck inside the attempt, and admit its failures to repair.

Until now the first thing to typecheck a change was the Reviewer, which runs *after* the
Engineer has already declared itself `completed`. So a type error -- however localized,
however visible in the file the attempt had just written -- was structurally guaranteed to
cost a full outer attempt and a 15-40 minute re-implementation. The exit condition for
`completed` was *lint passed*, and nothing between the model's last token and the Reviewer
ever asked whether the types agreed.

This is the clean half of that problem, and it is clean for one reason: **a type error is
unambiguous in a way a failing test is not**. A failing test may be broken scaffolding or a
correct test rejecting the implementation, which is why admitting one needed the assertion
guard in `tools.scoped_tests`. A typechecker rejecting a change is the checkout's own
compiler saying the code does not fit together, and there is no assertion to weaken and no
judgement to smuggle: the repair either makes the types agree or it does not.

Nothing here is repository-specific. The commands come from `validation_plan()`, which reads
them off the checkout's own manifests -- a `typecheck` script, `mypy`, `pyright` -- and a
repository that configures none gets nothing run and nothing claimed. Only the
typecheck-typed commands are run: lint is already covered by the source gate before this,
and the change's own tests come after it.

Two properties this shares with the narrowed test runner, for the same reasons:

* **Deliberately unjournaled.** `WorkspaceValidationTools` keys a journaled validation on the
  working-tree revision and the command, and the Reviewer runs the very same command a few
  minutes later -- so an attempt that needed no repair would produce an identical key and hand
  the Reviewer this run's verdict instead of its own. Fast feedback inside the attempt must
  never become the authority outside it.
* **A run that could not complete is not a verdict.** A typecheck that timed out or exhausted
  the machine says nothing about the change, and admitting one would spend a repair pass on a
  problem no rewrite can fix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from services.cancellation import CancellationToken, MockCancellationToken
from services.process_runner import ProcessRunner
from tools.file_tools import PathLike
from tools.validation_tools import (
    ValidationPlanBuilder,
    ValidationResult,
    WorkspaceValidationTools,
    rejects_the_source,
)

# How every diagnostic this module composes begins. The Engineer's repair loop matches on this
# constant rather than on a copy of the sentence, so the producer of a diagnostic and the
# predicate that admits it cannot drift apart. That is the whole complaint against the two
# bare string literals the predicate started with, and the reason widening it to typecheck
# output is done by tagging at the source rather than by adding a third literal.
TYPECHECK_DIAGNOSTIC_PREFIX = "The change does not typecheck"


@dataclass(frozen=True, slots=True)
class TypecheckOutcome:
    """What the checkout's own typecheck did, and what it has to say about the source.

    ``commands`` is recorded even when nothing failed, because "this repository configures no
    typecheck" and "it ran and passed" are different facts about an attempt and a reader
    should not have to infer which one happened.
    """

    commands: tuple[str, ...] = ()
    diagnostics: tuple[str, ...] = ()


class TypecheckRunner(Protocol):
    """Ask the repository's own typechecker about the workspace one attempt just wrote."""

    async def run_typecheck(self, workspace_root: PathLike) -> TypecheckOutcome:
        """Return what ran and what it reported; never raise for an ordinary failure."""


class NullTypecheckRunner:
    """Run nothing, which is what a composition with no configured runner must do."""

    async def run_typecheck(self, workspace_root: PathLike) -> TypecheckOutcome:
        """Report that no typecheck ran here."""
        del workspace_root
        return TypecheckOutcome()


class RepositoryTypecheckRunner:
    """Run the checkout's own typecheck commands against the attempt's workspace."""

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

    async def run_typecheck(self, workspace_root: PathLike) -> TypecheckOutcome:
        """Run every typecheck command this repository configures, and report its rejections."""
        tools = WorkspaceValidationTools(
            workspace_root,
            repository_id=self._repository_id,
            plan_builder=self._plan_builder,
            cancellation_token=self._cancellation_token,
            process_runner=self._process_runner,
            # No operation executor: see the module docstring.
            operation_executor=None,
        )
        results = await tools.run_typecheck_checks_async(timeout_seconds=self._timeout)
        return TypecheckOutcome(
            commands=tuple(" ".join(result.command) for result in results),
            diagnostics=tuple(
                typecheck_diagnostic(result) for result in results if rejects_the_source(result)
            ),
        )


def typecheck_diagnostic(result: ValidationResult) -> str:
    """Compose one repair-loop diagnostic from a typecheck run that rejected the source.

    The bounded summaries rather than the raw streams: they are already redacted, already
    length-bounded, and already carry the `[workspace files named by this output]` section
    that lets the repair prompt quote the source around each rejected location.
    """
    headline = (
        f"{TYPECHECK_DIAGNOSTIC_PREFIX} (command={' '.join(result.command)}; "
        f"code={result.result_code})."
    )
    excerpt = "\n".join(part for part in (result.stdout_summary, result.stderr_summary) if part)
    return f"{headline}\n{excerpt}" if excerpt.strip() else headline


def is_typecheck_diagnostic(diagnostic: str) -> bool:
    """Say whether one diagnostic is this module's, and therefore about the types."""
    return diagnostic.startswith(TYPECHECK_DIAGNOSTIC_PREFIX)


__all__ = [
    "TYPECHECK_DIAGNOSTIC_PREFIX",
    "NullTypecheckRunner",
    "RepositoryTypecheckRunner",
    "TypecheckOutcome",
    "TypecheckRunner",
    "is_typecheck_diagnostic",
    "typecheck_diagnostic",
]
