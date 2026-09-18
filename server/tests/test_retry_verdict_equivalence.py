"""The retry verdict has one producer, and it produces exactly what the old two produced.

Task `30-` moved three convergence guards out of ``_run_one_child`` and into
``decide_child_retry``. The guards ran *after* that function had already returned
``should_retry=True``: they stopped the workstream anyway and rewrote its reason, which is how
six production rows came to record their own stop as "Attempt N may proceed with a changed
strategy". Moving them changes where the answer is produced and must change nothing about
what the answer is.

The load-bearing artifact is ``_reference_verdict`` below: a transcription of the pre-refactor
code -- the caller's repeat clamp, the four refusals inside ``decide_child_retry``, and the
caller's two convergence guards -- in the order the platform applied them when they were
spread across two files. Every scenario in the table is decided twice, once by the
transcription and once by the function, and the two must agree on the verdict, on the reason,
and on what the attempt is recorded as having achieved.

The transcription is the *old* rule set. Nothing in it may be "corrected": if it and the
implementation disagree, the implementation changed behaviour and that is the defect this
module exists to catch.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from typing import NamedTuple

import pytest

from tools.retry_strategy import (
    FailureClassification,
    RetryDecision,
    decide_child_retry,
    retry_counter_field,
)

# ---------------------------------------------------------------------------------------
# The reason strings, quoted from the pre-refactor source rather than imported
# ---------------------------------------------------------------------------------------
#
# Deliberately literal. Importing the constants the implementation uses would make this table
# agree with any rewording, which is the one thing it exists to notice: a reason string is
# what an operator reads off a dead workstream, and `25-` exists because those strings were
# not trustworthy.

_REPRODUCED_PREVIOUS_CHANGE = "attempt_reproduced_the_previous_production_change"
_RETURNED_TO_EARLIER_STATE = "attempt_returned_to_an_earlier_submitted_state"

_SETUP_REFUSAL = (
    "The repository's checked-in setup is invalid; an approved repository repair is required "
    "before coding can be retried."
)
_NO_MEANINGFUL_CHANGE_REFUSAL = (
    "The previous attempt produced no meaningful production change, so repeating it would "
    "submit identical inputs."
)
_NON_CONVERGING_REFUSAL = (
    "The same diagnostics came back unchanged, so this workstream is not converging and its "
    "remaining attempts were not spent."
)

# The three bounds, quoted from where they lived in `workflows/feature_workflow.py` before
# this task moved them. Changing any of them is out of scope for the refactor, so the numbers
# here are also an assertion that they did not move.
_ALLOWED_IDENTICAL_RESUBMISSIONS = 2
_ALLOWED_RETURNS_TO_EARLIER_STATES = 2
_ALLOWED_REPEATED_DIAGNOSTIC_SIGNATURES = 3

_TERMINAL_SETUP_CLASSIFICATIONS = {
    FailureClassification.VALIDATION_CONFIGURATION_FAILURE,
    FailureClassification.TEST_INFRASTRUCTURE_MISSING,
}
_CAPACITY_CLASSIFICATIONS = {FailureClassification.VALIDATION_CAPACITY_FAILURE}


class _Scenario(NamedTuple):
    """One row of the table: everything the loop knew when it asked for a verdict."""

    classification: FailureClassification
    attempt_count: int
    budget: int
    retry_count: int
    max_child_review_cycles: int
    meaningful_change: bool
    meaningful_change_reason: str
    production_change_present: bool
    revisited_earlier_state: bool
    source_rejected_before_commit: bool
    findings_repeat_previous: bool
    diagnostic_signature_repeated: bool
    deterministic_gate: bool
    identical_resubmissions: int
    returns_to_earlier_states: int
    repeated_diagnostic_signatures: int


class _Outcome(NamedTuple):
    """What the platform decided, and what it recorded about the attempt it judged."""

    should_retry: bool
    reason: str
    meaningful_change: bool
    meaningful_change_reason: str


def _reference_verdict(scenario: _Scenario) -> _Outcome:
    """Decide one scenario the way the platform decided it before this task.

    Transcribed in three parts, in execution order:

    1. ``_run_one_child``'s repeat clamp, which ran before the call and could overturn
       ``meaningful_progress``'s verdict and rewrite its reason;
    2. ``decide_child_retry``'s four refusals, unchanged;
    3. ``_run_one_child``'s two convergence guards, which ran *after* the call returned
       ``should_retry=True`` and could stop the workstream regardless.
    """
    identical = scenario.identical_resubmissions
    returns = scenario.returns_to_earlier_states
    meaningful = scenario.meaningful_change
    reason = scenario.meaningful_change_reason

    # -- 1. the repeat clamp -------------------------------------------------------------
    circling_is_unknowable = scenario.source_rejected_before_commit
    if (
        reason == _REPRODUCED_PREVIOUS_CHANGE
        or (scenario.revisited_earlier_state and not circling_is_unknowable)
    ) and scenario.production_change_present:
        if scenario.revisited_earlier_state and not circling_is_unknowable:
            reason = _RETURNED_TO_EARLIER_STATE
            returns += 1
        identical += 1
        meaningful = (
            identical < _ALLOWED_IDENTICAL_RESUBMISSIONS
            and returns < _ALLOWED_RETURNS_TO_EARLIER_STATES
        )
    elif not circling_is_unknowable:
        identical = 0

    # -- 2. the four refusals ------------------------------------------------------------
    counter_field = retry_counter_field(scenario.classification)
    if scenario.classification in _TERMINAL_SETUP_CLASSIFICATIONS:
        return _Outcome(False, _SETUP_REFUSAL, meaningful, reason)
    if (
        scenario.classification is not FailureClassification.DEPENDENCY_INSTALLATION_FAILURE
        and scenario.classification not in _CAPACITY_CLASSIFICATIONS
        and not meaningful
    ):
        return _Outcome(False, _NO_MEANINGFUL_CHANGE_REFUSAL, meaningful, reason)
    if scenario.attempt_count >= scenario.budget:
        return _Outcome(
            False,
            f"The {counter_field} budget of {scenario.budget} is exhausted.",
            meaningful,
            reason,
        )
    if scenario.retry_count + 1 >= scenario.max_child_review_cycles:
        return _Outcome(
            False,
            (f"The child review cycle limit of {scenario.max_child_review_cycles} is reached."),
            meaningful,
            reason,
        )

    # -- 3. the convergence guards that used to run after the verdict --------------------
    repeated_model_findings = scenario.findings_repeat_previous and not scenario.deterministic_gate
    signatures = (
        scenario.repeated_diagnostic_signatures + 1 if scenario.diagnostic_signature_repeated else 0
    )
    if not scenario.deterministic_gate and signatures >= (
        _ALLOWED_REPEATED_DIAGNOSTIC_SIGNATURES - 1
    ):
        repeated_model_findings = True
    if repeated_model_findings:
        return _Outcome(False, _NON_CONVERGING_REFUSAL, meaningful, reason)
    return _Outcome(
        True,
        f"Attempt {scenario.retry_count + 1} may proceed with a changed strategy.",
        meaningful,
        reason,
    )


def _decide(scenario: _Scenario) -> RetryDecision:
    """Ask the single authority the same question the table asked the transcription."""
    return decide_child_retry(
        classification=scenario.classification,
        attempt_count=scenario.attempt_count,
        budget=scenario.budget,
        retry_count=scenario.retry_count,
        max_child_review_cycles=scenario.max_child_review_cycles,
        meaningful_change=scenario.meaningful_change,
        meaningful_change_reason=scenario.meaningful_change_reason,
        production_change_present=scenario.production_change_present,
        revisited_earlier_state=scenario.revisited_earlier_state,
        source_rejected_before_commit=scenario.source_rejected_before_commit,
        findings_repeat_previous=scenario.findings_repeat_previous,
        diagnostic_signature_repeated=scenario.diagnostic_signature_repeated,
        deterministic_gate=scenario.deterministic_gate,
        identical_resubmissions=scenario.identical_resubmissions,
        returns_to_earlier_states=scenario.returns_to_earlier_states,
        repeated_diagnostic_signatures=scenario.repeated_diagnostic_signatures,
    )


# The table's axes. Two values per binary dimension and two per bounded counter -- one below
# its allowance and one at it -- because every rule in the verdict is a threshold and a
# threshold is exercised by the value under it and the value on it.
_BUDGETS = ((0, 4), (4, 4))
_CYCLES = ((0, 5), (4, 5))
_PROGRESS_REASONS = (
    _REPRODUCED_PREVIOUS_CHANGE,
    "new_or_material_production_source_implementation",
)


_FLAGS: tuple[bool, ...] = (True, False)
_CARRIED: tuple[int, ...] = (0, 1)


def _table() -> Iterator[_Scenario]:
    """Every combination of classification, budget state, and convergence evidence."""
    for classification in FailureClassification:
        for (attempt_count, budget), (retry_count, cycles) in itertools.product(_BUDGETS, _CYCLES):
            for reason in _PROGRESS_REASONS:
                for flags in itertools.product(_FLAGS, repeat=7):
                    meaningful, production, revisited, rejected, repeat, signature, gate = flags
                    for carried in itertools.product(_CARRIED, repeat=3):
                        identical, returns, signatures = carried
                        yield _Scenario(
                            classification=classification,
                            attempt_count=attempt_count,
                            budget=budget,
                            retry_count=retry_count,
                            max_child_review_cycles=cycles,
                            meaningful_change=meaningful,
                            meaningful_change_reason=reason,
                            production_change_present=production,
                            revisited_earlier_state=revisited,
                            source_rejected_before_commit=rejected,
                            findings_repeat_previous=repeat,
                            diagnostic_signature_repeated=signature,
                            deterministic_gate=gate,
                            identical_resubmissions=identical,
                            returns_to_earlier_states=returns,
                            repeated_diagnostic_signatures=signatures,
                        )


_TABLE = tuple(_table())


def test_the_scenario_table_covers_every_axis_it_claims_to() -> None:
    """A table that silently shrank would report equivalence it never checked."""
    # 8 classifications x 2 budget states x 2 cycle states x 11 binary or two-valued
    # dimensions.
    assert len(_TABLE) == 8 * 2 * 2 * 2**11
    assert len({item.classification for item in _TABLE}) == len(FailureClassification)


def test_every_scenario_reaches_the_same_verdict_and_the_same_reason() -> None:
    """The whole point of the refactor: one producer, and the same answers it replaced.

    Compared field by field rather than as a whole object, so a failure names which of the
    three -- the verdict, the sentence recorded as the stop, or what the attempt is recorded
    as having achieved -- moved.
    """
    disagreements: list[str] = []
    for scenario in _TABLE:
        expected = _reference_verdict(scenario)
        decision = _decide(scenario)
        actual = _Outcome(
            decision.should_retry,
            decision.reason,
            decision.meaningful_change,
            decision.meaningful_change_reason,
        )
        if actual != expected:
            disagreements.append(f"{scenario}\n  expected {expected}\n  got      {actual}")
    assert not disagreements, (
        f"{len(disagreements)} of {len(_TABLE)} scenarios changed behaviour:\n"
        + "\n".join(disagreements[:5])
    )


def test_the_table_actually_exercises_every_verdict_the_platform_can_reach() -> None:
    """A table that only ever produced one answer would pass and prove nothing."""
    reasons = {_reference_verdict(scenario).reason for scenario in _TABLE}
    budget_refusals = {reason for reason in reasons if "budget of" in reason}
    cycle_refusals = {reason for reason in reasons if "review cycle limit" in reason}
    permitted = {reason for reason in reasons if "may proceed with a changed strategy" in reason}
    assert _SETUP_REFUSAL in reasons
    assert _NO_MEANINGFUL_CHANGE_REFUSAL in reasons
    assert _NON_CONVERGING_REFUSAL in reasons
    # One per counter the classifications map onto, and one per configured limit.
    assert len(budget_refusals) == 4
    assert cycle_refusals == {"The child review cycle limit of 5 is reached."}
    assert permitted == {"Attempt 1 may proceed with a changed strategy."}


def test_the_counter_field_a_refusal_names_never_changed() -> None:
    """A refusal that named the wrong budget would send its reader to the wrong number."""
    for scenario in _TABLE:
        assert _decide(scenario).counter_field == retry_counter_field(scenario.classification)


def test_a_permitted_attempt_still_asks_the_operator_nothing() -> None:
    """Work that may continue is not an escalation, and must not read like one."""
    permitted = [scenario for scenario in _TABLE if _decide(scenario).should_retry]
    assert permitted
    for scenario in permitted:
        decision = _decide(scenario)
        assert decision.operator_question == ""
        assert decision.non_converging is False


def test_every_refusal_except_the_convergence_stop_hands_over_a_question() -> None:
    """The convergence stop's empty question is the one this task deliberately preserved.

    It reached ``triage_stopped_workstream`` as a fallback that the triage never uses --
    ``blocker_repeated`` is true by construction there, so the triage answers from its own
    text. Filling it in would be an improvement, and an improvement is a behaviour change.
    """
    for scenario in _TABLE:
        decision = _decide(scenario)
        if decision.should_retry:
            continue
        if decision.non_converging:
            assert decision.operator_question == ""
            assert decision.reason == _NON_CONVERGING_REFUSAL
            continue
        assert decision.operator_question.endswith("?")


def test_a_stopped_workstream_reports_whether_its_blocker_stopped_responding() -> None:
    """``triage_stopped_workstream`` reads this, and the loop no longer computes it."""
    for scenario in _TABLE:
        decision = _decide(scenario)
        if decision.non_converging:
            assert decision.blocker_repeated is True
        else:
            assert decision.blocker_repeated is (decision.identical_resubmissions > 0)


# ---------------------------------------------------------------------------------------
# The named regressions. Each constant in the task's section 3.3 has an incident behind it.
# ---------------------------------------------------------------------------------------


def _repeating(**overrides: object) -> RetryDecision:
    """Ask for a verdict on an attempt whose budget and cycles are nowhere near spent."""
    arguments: dict[str, object] = {
        "classification": FailureClassification.VALIDATION_SOURCE_FAILURE,
        "attempt_count": 0,
        "budget": 8,
        "retry_count": 0,
        "max_child_review_cycles": 12,
        "meaningful_change": True,
        "production_change_present": True,
    }
    arguments.update(overrides)
    return decide_child_retry(**arguments)  # type: ignore[arg-type]


def test_a_workstream_repeating_once_is_allowed_a_second_attempt() -> None:
    """Repeating once is a model that did not read its diagnostic, not one that cannot.

    r-2's backend had registered its route and was one documentation edit from approval when
    it resent the same five files. Stopping on the first repeat spent two attempts of eight.
    """
    decision = _repeating(meaningful_change_reason=_REPRODUCED_PREVIOUS_CHANGE)

    assert decision.should_retry is True
    assert decision.identical_resubmissions == 1
    assert decision.meaningful_change is True


def test_a_workstream_repeating_twice_is_stopped() -> None:
    """The second repeat still ends it, so ser-1's ten wasted attempts cannot come back."""
    first = _repeating(meaningful_change_reason=_REPRODUCED_PREVIOUS_CHANGE)
    second = _repeating(
        meaningful_change_reason=_REPRODUCED_PREVIOUS_CHANGE,
        identical_resubmissions=first.identical_resubmissions,
    )

    assert second.should_retry is False
    assert second.reason == _NO_MEANINGFUL_CHANGE_REFUSAL
    assert second.meaningful_change is False
    assert second.identical_resubmissions == _ALLOWED_IDENTICAL_RESUBMISSIONS


def test_a_workstream_cycling_a_b_a_c_a_is_stopped() -> None:
    """The consecutive counter cannot see a circle with any variety in it.

    AB-Feature-124's console went 04199e3a -> 53188603 -> 04199e3a -> ac8cbe16 -> 04199e3a:
    three arrivals at the same state, each correctly detected, and each forgotten by the time
    the next one came, because the genuinely-new attempt in between reset the count. It ran
    to its ceiling regardless.
    """
    identical = 0
    returns = 0
    verdicts: list[bool] = []
    # A, B, A, C, A -- the two 'A's after the first are returns to a state already submitted;
    # 'B' and 'C' are new, and are what used to clear the consecutive count.
    for revisited in (False, False, True, False, True):
        decision = _repeating(
            meaningful_change_reason="new_or_material_production_source_implementation",
            revisited_earlier_state=revisited,
            identical_resubmissions=identical,
            returns_to_earlier_states=returns,
        )
        identical = decision.identical_resubmissions
        returns = decision.returns_to_earlier_states
        verdicts.append(decision.should_retry)

    assert verdicts == [True, True, True, True, False]
    assert returns == _ALLOWED_RETURNS_TO_EARLIER_STATES


def test_a_deterministic_gate_repeating_its_sentence_does_not_stop_the_workstream() -> None:
    """A check that repeats itself is consistent, not a model ignoring feedback.

    Feature -052b lost both workstreams two attempts in because the reachability gate said
    the same true thing twice while the attempts around it were changing. The loop-level
    proof is ``test_a_deterministic_gate_repeating_itself_still_gets_its_retries`` in
    ``tests/test_feature_workflow.py``; this is the same rule at the verdict.
    """
    modelled = _repeating(findings_repeat_previous=True, deterministic_gate=False)
    gated = _repeating(findings_repeat_previous=True, deterministic_gate=True)

    assert modelled.should_retry is False
    assert modelled.reason == _NON_CONVERGING_REFUSAL
    assert gated.should_retry is True


def test_three_identical_diagnostic_signatures_stop_it_and_two_do_not() -> None:
    """Three, not two, because reducing a diagnostic to its signature is lossy.

    rep-a's frontend was ended three attempts in while its source was being rewritten behind
    an unchanging linter message, so one extra attempt is the documented price of not doing
    that again.
    """
    signatures = 0
    verdicts: list[bool] = []
    # The first attempt has nothing to repeat; the second and third name the same defect.
    for repeated in (False, True, True):
        decision = _repeating(
            diagnostic_signature_repeated=repeated,
            repeated_diagnostic_signatures=signatures,
        )
        signatures = decision.repeated_diagnostic_signatures
        verdicts.append(decision.should_retry)

    assert verdicts == [True, True, False]
    assert signatures == _ALLOWED_REPEATED_DIAGNOSTIC_SIGNATURES - 1


def test_a_pre_commit_rejection_does_not_count_as_reproducing_the_previous_change() -> None:
    """The workspace was reset, so the fingerprint describes the start state.

    -076's console produced two attempts with byte-identical fingerprints that failed at
    different gates, which cannot both be true of the same source. Two exemptions follow, and
    the verdict owns both: a rejected attempt is not read as a return to an earlier state,
    and it does not clear a count of repeats it says nothing about. Clearing it is how -112's
    console spent all twelve review cycles submitting five copies of one production diff.
    ``test_a_precommit_rejection_is_progress_only_when_something_changed`` in
    ``tests/test_repository_preflight_and_retry.py`` covers the same flag inside
    ``meaningful_progress``.
    """
    circling = _repeating(revisited_earlier_state=True, source_rejected_before_commit=True)
    assert circling.should_retry is True
    assert circling.returns_to_earlier_states == 0

    preserved = _repeating(
        meaningful_change_reason="new_or_material_production_source_implementation",
        source_rejected_before_commit=True,
        identical_resubmissions=1,
    )
    assert preserved.identical_resubmissions == 1

    cleared = _repeating(
        meaningful_change_reason="new_or_material_production_source_implementation",
        source_rejected_before_commit=False,
        identical_resubmissions=1,
    )
    assert cleared.identical_resubmissions == 0


@pytest.mark.parametrize(
    "classification",
    [
        FailureClassification.DEPENDENCY_INSTALLATION_FAILURE,
        FailureClassification.VALIDATION_CAPACITY_FAILURE,
    ],
)
def test_the_exempt_routes_stay_exempt_from_the_meaningful_change_rule(
    classification: FailureClassification,
) -> None:
    """The change these ask for is not to application source, so no diff can be demanded."""
    decision = _repeating(classification=classification, meaningful_change=False)
    assert decision.should_retry is True
