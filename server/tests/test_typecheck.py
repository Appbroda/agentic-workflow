"""The repository's own typecheck, asked inside the attempt.

The end-to-end proof lives in the `realrepo` tier, against a real checkout whose typecheck is
a real program (`test_real_repository_gates.py`, B1 and B2). These are the propositions
underneath it, in the default tier: which commands this is allowed to run, what it will hand
to a repair loop, and -- the one that matters most -- what it refuses to hand over.

That refusal is the whole safety case for admitting a new check to the loop. A run that timed
out or exhausted the machine has said nothing about the change, so a repair pass spent on one
is a coding budget spent on a machine problem. It is the same rule the narrowed test runner
holds itself to, and it is now literally the same predicate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from services.process_runner import ProcessResult
from tests.fixtures import node_javascript_repository
from tools.technology_detection import inspect_repository_technology
from tools.typecheck import (
    TYPECHECK_DIAGNOSTIC_PREFIX,
    NullTypecheckRunner,
    RepositoryTypecheckRunner,
    is_typecheck_diagnostic,
)
from tools.validation_tools import (
    MockValidationPlanBuilder,
    RepositoryValidationPlan,
    ValidationCommand,
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
    """A repository declaring a linter, a typechecker and a test command."""
    return RepositoryValidationPlan(
        repository_id="web",
        technology_profile=inspect_repository_technology(root, repository_id="web"),
        commands=[
            ValidationCommand(
                validation_type="lint", command=["npm", "run", "lint"], timeout_seconds=60
            ),
            ValidationCommand(
                validation_type="typecheck",
                command=["npm", "run", "typecheck"],
                timeout_seconds=60,
            ),
            ValidationCommand(
                validation_type="test", command=["npm", "run", "test"], timeout_seconds=60
            ),
        ],
        source="repository_scripts",
    )


@pytest.mark.asyncio
async def test_only_the_typecheck_commands_run_and_nothing_else_does(tmp_path: Path) -> None:
    """Lint is already covered by the source gate, and a whole test run costs the minutes
    the narrowing exists to avoid.

    Running the full plan here would ask the same questions the gate before it and the runner
    after it are already asking, twice per attempt, for verdicts the reviewer then produces
    again. So the type is the selector, and it is read off the checkout's own plan rather than
    from anything this module knows about a repository.
    """
    root = node_javascript_repository(tmp_path / "web")
    runner = _ScriptedRunner(
        ProcessResult(command=(), return_code=0, stdout="", stderr="", duration_seconds=0.1)
    )

    outcome = await RepositoryTypecheckRunner(
        process_runner=runner, plan_builder=MockValidationPlanBuilder(_plan(root))
    ).run_typecheck(root)

    assert runner.commands == [("npm", "run", "typecheck")]
    assert outcome.commands == ("npm run typecheck",)
    assert outcome.diagnostics == ()


@pytest.mark.asyncio
async def test_a_rejected_typecheck_becomes_one_tagged_diagnostic(tmp_path: Path) -> None:
    """The diagnostic carries the tag its own producer defines, not a sentence copied twice.

    That is the point of widening the repair loop's predicate by tagging at the source: the
    thing that composes a diagnostic and the thing that admits one now share a constant, so
    they cannot drift apart the way a third bare string literal in a prefix tuple would.
    """
    root = node_javascript_repository(tmp_path / "web")
    runner = _ScriptedRunner(
        ProcessResult(
            command=(),
            return_code=2,
            stdout=(
                "src/routes/metrics.js(1,1): error TS2322: Type 'string' is not assignable "
                "to type 'number'.\n"
            ),
            stderr="",
            duration_seconds=0.4,
        )
    )

    outcome = await RepositoryTypecheckRunner(
        process_runner=runner, plan_builder=MockValidationPlanBuilder(_plan(root))
    ).run_typecheck(root)

    assert len(outcome.diagnostics) == 1
    assert is_typecheck_diagnostic(outcome.diagnostics[0])
    assert outcome.diagnostics[0].startswith(TYPECHECK_DIAGNOSTIC_PREFIX)
    assert "TS2322" in outcome.diagnostics[0]
    assert "npm run typecheck" in outcome.diagnostics[0]


@pytest.mark.asyncio
async def test_a_typecheck_that_could_not_complete_says_nothing_about_the_source(
    tmp_path: Path,
) -> None:
    """B2's other half: a run with no verdict is not admitted to the repair loop at all.

    A typecheck that timed out has not rejected the change -- it has not judged it. Rewriting
    the file cannot make a command fit inside its budget, so admitting this would put a repair
    pass to work on a question nobody answered. The command is still recorded, because "it ran
    and could not finish" is a fact worth keeping.
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

    outcome = await RepositoryTypecheckRunner(
        process_runner=runner, plan_builder=MockValidationPlanBuilder(_plan(root))
    ).run_typecheck(root)

    assert outcome.commands == ("npm run typecheck",)
    assert outcome.diagnostics == ()


@pytest.mark.asyncio
async def test_a_repository_that_configures_no_typecheck_runs_nothing(tmp_path: Path) -> None:
    """No configured command, no subprocess, and no claim that the types were checked."""
    root = node_javascript_repository(tmp_path / "web")
    runner = _ScriptedRunner(
        ProcessResult(command=(), return_code=0, stdout="", stderr="", duration_seconds=0.1)
    )
    plan = _plan(root)
    without_typecheck = plan.model_copy(
        update={
            "commands": [
                command for command in plan.commands if command.validation_type != "typecheck"
            ]
        }
    )

    outcome = await RepositoryTypecheckRunner(
        process_runner=runner, plan_builder=MockValidationPlanBuilder(without_typecheck)
    ).run_typecheck(root)

    assert runner.commands == []
    assert outcome == await NullTypecheckRunner().run_typecheck(root)
