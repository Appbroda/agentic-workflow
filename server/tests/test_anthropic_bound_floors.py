"""Startup refuses an Anthropic output bound its own configured effort cannot finish under.

Thinking spends the same ``max_tokens`` bound the answer does, so a bound sized only for the
answer truncates exactly when the effort is doing its job -- and a truncation is deterministic
(`response_truncated`, failed once, never retried), so it is terminal for whatever stage hits
it. The AB-Feature-175-180 matrix hit it at every Anthropic stage that ran: two planners at
32 000 under `max`, a reviewer at 16 000 under `high`, and a coding call. Every one of those
was checkable before any feature existed, which is what these tests hold: the exact
configurations that killed those runs cannot start.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from configs.settings import CONFIG_DIRECTORY, load_settings
from tests.test_anthropic_adapter import anthropic_settings
from tests.test_performance_tiers import _cleared_tier_fields


def test_the_review_bound_that_killed_180_is_refused_at_startup() -> None:
    """AB-Feature-180's reviewer: effort `high` against a 16 000 bound, terminal at attempt 0.

    The error must name the variable whose value was read and the floor it missed, because
    the variable name is exactly what a deployment edits.
    """
    with pytest.raises(ValidationError, match="ANTHROPIC_REVIEW_MAX_TOKENS") as refused:
        anthropic_settings(anthropic_review_max_tokens=16_000)

    text = str(refused.value)
    assert "32000" in text  # the floor for effort `high`
    assert "ANTHROPIC_REVIEW_REASONING_EFFORT" in text


def test_the_reasoning_bound_that_killed_175_is_refused_at_startup() -> None:
    """AB-Feature-175's planner: effort `max` against a 32 000 bound, truncated twice."""
    with pytest.raises(ValidationError, match="ANTHROPIC_REASONING_MAX_TOKENS"):
        anthropic_settings(anthropic_reasoning_max_tokens=32_000)


def test_a_tier_bound_below_its_own_efforts_floor_names_the_tier_variable() -> None:
    """A tier that declares its own too-small bound is blamed by its own suffixed name."""
    with pytest.raises(ValidationError, match="ANTHROPIC_MEDIUM_REVIEW_MAX_TOKENS"):
        anthropic_settings(
            anthropic_medium_reasoning_model="claude-opus-5",
            anthropic_medium_reasoning_effort="high",
            anthropic_medium_coding_model="claude-sonnet-5",
            anthropic_medium_coding_reasoning_effort="xhigh",
            anthropic_medium_coding_max_tokens=128_000,
            anthropic_medium_review_model="claude-sonnet-5",
            anthropic_medium_review_reasoning_effort="high",
            anthropic_medium_review_max_tokens=16_000,
            anthropic_medium_scoped_fix_model="claude-sonnet-5",
            anthropic_medium_scoped_fix_reasoning_effort="high",
        )


def test_a_tier_inheriting_a_too_small_bound_names_the_unsuffixed_variable() -> None:
    """The 175-180 configuration exactly: tiers set no bounds and inherited the platform's.

    The tier's review runs at `high` while the unsuffixed review is configured legal at
    `low`, so only the *tier's* resolution breaks -- and the variable named must be the
    unsuffixed one, because that is where the too-small value actually lives.
    """
    with pytest.raises(ValidationError, match="ANTHROPIC_REVIEW_MAX_TOKENS") as refused:
        anthropic_settings(
            anthropic_review_reasoning_effort="low",
            anthropic_review_max_tokens=16_000,
            anthropic_medium_reasoning_model="claude-opus-5",
            anthropic_medium_reasoning_effort="high",
            anthropic_medium_coding_model="claude-sonnet-5",
            anthropic_medium_coding_reasoning_effort="xhigh",
            anthropic_medium_coding_max_tokens=128_000,
            anthropic_medium_review_model="claude-sonnet-5",
            anthropic_medium_review_reasoning_effort="high",
            anthropic_medium_scoped_fix_model="claude-sonnet-5",
            anthropic_medium_scoped_fix_reasoning_effort="high",
            anthropic_medium_scoped_fix_max_tokens=32_000,
        )

    assert "ANTHROPIC_MEDIUM_REVIEW_MAX_TOKENS" not in str(refused.value)


def test_the_shipped_defaults_clear_every_floor_for_the_recommended_efforts() -> None:
    """The defaults hold at the most demanding efforts a deployment is documented to run."""
    settings = anthropic_settings()

    assert settings.anthropic_reasoning_max_tokens >= 64_000
    assert settings.anthropic_coding_max_tokens >= 128_000
    assert settings.anthropic_review_max_tokens >= 32_000
    assert settings.anthropic_scoped_fix_max_tokens >= 32_000


def test_an_unconfigured_role_is_not_held_to_a_floor() -> None:
    """A role no model executes has no call to truncate, so its bound is never read.

    An OpenAI-only deployment leaves every Anthropic model unset while the bound fields keep
    their defaults; validating those would refuse a configuration that makes no Anthropic
    request at all.
    """
    settings = load_settings(
        CONFIG_DIRECTORY,
        # The deployment `.env` names Anthropic tier presets whose bounds fall back to the
        # 1_000 below; pinned out so this test decides its own configuration.
        **_cleared_tier_fields(),
        anthropic_reasoning_model="",
        anthropic_coding_model="",
        anthropic_review_model="",
        anthropic_scoped_fix_model="",
        anthropic_reasoning_effort="max",
        anthropic_reasoning_max_tokens=1_000,
    )

    assert settings.anthropic_reasoning_max_tokens == 1_000


def test_an_unset_effort_is_still_held_to_the_adaptive_default_floor() -> None:
    """Unset effort sends no thinking configuration, and the provider default still thinks."""
    with pytest.raises(ValidationError, match="ANTHROPIC_SCOPED_FIX_MAX_TOKENS"):
        anthropic_settings(
            anthropic_scoped_fix_reasoning_effort="",
            anthropic_scoped_fix_max_tokens=8_000,
        )
