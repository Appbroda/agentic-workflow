"""The canary, run by the suite so it cannot rot between the nights it is meant to run.

A canary is only worth having if it still works when something finally breaks, and the way
that stops being true is nobody running it for a month. This is not a second copy of the
invariants -- `tests/canary.py` owns those -- it is the check that the runnable target still
composes, still produces machine-readable output, and still reports every invariant it
claims to.

Marked `realrepo` because it executes a repository's own toolchain, and it additionally needs
a PostgreSQL server to create its own database on. When either is absent it skips with the
reason, exactly as the other two tiers do.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from tests.canary import CanaryReport, main, run_canary
from tests.fixtures.real_repositories import (
    missing_executables,
    requires_real_repository_tier,
    unavailable_reason,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.realrepo]

_EXPECTED_INVARIANTS = {
    "every_approved_child_has_a_pull_request",
    "no_child_left_running_under_a_terminal_feature",
    "every_terminal_feature_is_classified_and_diagnosed",
    "every_attempted_workstream_has_a_current_revision",
    "no_duplicate_clone_commit_push_or_pull_request",
    "no_feature_exceeds_its_runtime_ceiling_without_a_terminal_state",
    "every_planning_call_is_journaled",
    "every_routing_record_names_the_tier",
}


@pytest.fixture(autouse=True)
def _toolchain() -> None:
    """Refuse to certify anything on a machine that cannot run these checkouts."""
    missing = missing_executables("git", "node", "npm")
    if not missing:
        return
    if requires_real_repository_tier():
        pytest.fail(unavailable_reason(missing), pytrace=False)
    pytest.skip(unavailable_reason(missing))


def _skip_if_it_could_not_run(report: CanaryReport) -> None:
    """A canary that could not run is not a canary that passed, and must not read as one."""
    if report.error is None:
        return
    if requires_real_repository_tier():
        pytest.fail(report.error, pytrace=False)
    pytest.skip(report.error)


def _scenario(report: CanaryReport, feature_id: str) -> dict[str, Any]:
    """Return one scenario summary by the feature it drove."""
    return next(item for item in report.scenarios if item["feature_id"] == feature_id)


async def test_the_canary_runs_three_real_features_and_answers_every_invariant(
    tmp_path: Path,
) -> None:
    """The whole target, end to end, on real checkouts and a database of its own.

    All three shapes matter. A canary that only ever published would never exercise the
    invariants about a feature that stopped -- which is where every one of the production
    violations behind them actually happened -- and one whose reviewer always approved could
    not tell the two positions of the review-scope switch apart.
    """
    report = await run_canary(tmp_path)
    _skip_if_it_could_not_run(report)

    assert {item.name for item in report.invariants} == _EXPECTED_INVARIANTS
    failing = [item for item in report.invariants if not item.passed]
    assert failing == [], f"the canary found an invariant violation: {failing}"
    assert report.passed
    assert report.bounded_review_scope is False

    published = _scenario(report, "canary-published")
    stopped = _scenario(report, "canary-stopped")
    unscoped = _scenario(report, "canary-unscoped-review")
    assert published["status"] == "completed"
    assert set(published["children"].values()) == {"completed"}
    assert stopped["status"] == "failed_requires_human"
    assert set(stopped["children"].values()) == {"failed"}
    # The default: an unscoped finding blocks, spends the workstream's attempts, and the
    # feature ends needing a person with nothing published.
    assert unscoped["status"] == "failed_requires_human"
    assert set(unscoped["children"].values()) == {"failed"}
    assert unscoped["review_findings"] == {
        "total": 0,
        "blocking": 0,
        "advisory": 0,
        "untraceable": 0,
    }
    assert unscoped["attempts"]["backend"] > 1
    # Every invariant cites the production count that made it one, so a future failure reads
    # as a regression against a known number rather than as an unexplained assertion.
    assert all(item.production_baseline for item in report.invariants)


async def test_the_canary_answers_the_same_invariants_with_review_scope_bounded(
    tmp_path: Path,
) -> None:
    """The other position of the switch, on the same fixtures.

    This is the comparison the behaviour change is judged on. Two things must both hold: the
    unscoped rejection now publishes and is recorded as advisory, and every invariant that
    held with the switch off still holds -- a completion rate bought by losing work is not
    an improvement.
    """
    report = await run_canary(tmp_path, bounded_review_scope=True)
    _skip_if_it_could_not_run(report)

    assert report.bounded_review_scope is True
    failing = [item for item in report.invariants if not item.passed]
    assert failing == [], f"the canary found an invariant violation: {failing}"

    # Unchanged: neither of these features turns on a review finding at all.
    assert _scenario(report, "canary-published")["status"] == "completed"
    assert _scenario(report, "canary-stopped")["status"] == "failed_requires_human"
    # Changed, and this is the whole measurement: the same review, the same code, published
    # with its finding carried to a person instead of spending the attempt budget.
    unscoped = _scenario(report, "canary-unscoped-review")
    assert unscoped["status"] == "completed"
    assert set(unscoped["children"].values()) == {"completed"}
    assert unscoped["review_findings"] == {
        "total": 1,
        "blocking": 0,
        "advisory": 1,
        "untraceable": 1,
    }
    # And it cost one attempt, not the several the same finding bought with the switch off.
    assert unscoped["attempts"]["backend"] == 1


async def test_the_canary_writes_a_summary_a_job_can_read_without_a_log(
    tmp_path: Path,
) -> None:
    """Machine-readable output is the whole reporting contract, so it is asserted as one.

    Driven through the command-line entry point in a worker thread, because that entry point
    owns its own event loop -- which is the thing a scheduled job actually invokes.
    """
    destination = tmp_path / "canary.json"

    code = await asyncio.to_thread(main, ["--json", str(destination), "--quiet"])

    if code == 2:  # pragma: no cover - only when the environment cannot run the canary
        _skip_if_it_could_not_run(CanaryReport(error=json.loads(destination.read_text())["error"]))
    assert code == 0, destination.read_text(encoding="utf-8")
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["error"] is None
    assert {item["name"] for item in payload["invariants"]} == _EXPECTED_INVARIANTS
    for item in payload["invariants"]:
        assert set(item) == {"name", "passed", "observed", "detail", "production_baseline"}
        assert item["passed"] is True
        assert item["observed"] == 0
