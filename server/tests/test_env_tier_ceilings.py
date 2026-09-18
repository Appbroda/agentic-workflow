"""Startup holds the environment-backed tiers to the ceiling rules custom setups pass.

00-todo item 29, shipped by 74-: `validate_model_setup` refused the AB-Feature-181 pairing
for a user-authored setup while the deployment's own env tiers were never checked, so the
original 181 vector stayed expressible in `.env` and failed only at runtime, mid-feature.
The rule is now one function (`judge_anthropic_ceiling`) with two policies: the authoring
path refuses everything refusable, and startup refuses only a pairing no bound value could
ever satisfy -- everything fixable becomes a named warning, because refusing startup can
take down a running deployment over configuration that predates the declaration.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from configs.model_roles import (
    ModelConfigurationError,
    parse_model_setup_roles,
    validate_model_setup,
)
from tests.test_anthropic_adapter import anthropic_settings

# A model name no other fixture uses, so the declared ceiling touches only the tier under
# test and never the unsuffixed defaults beside it.
TIER_MODEL = "claude-tier-probe"


def _medium_tier(**overrides: object) -> dict[str, object]:
    """A complete medium tier (a partial one is refused at startup by the missing name)."""
    values: dict[str, object] = {
        "anthropic_medium_reasoning_model": TIER_MODEL,
        "anthropic_medium_reasoning_effort": "high",
        "anthropic_medium_reasoning_max_tokens": 48_000,
        "anthropic_medium_coding_model": TIER_MODEL,
        "anthropic_medium_coding_reasoning_effort": "high",
        "anthropic_medium_coding_max_tokens": 48_000,
        "anthropic_medium_review_model": TIER_MODEL,
        "anthropic_medium_review_reasoning_effort": "high",
        "anthropic_medium_review_max_tokens": 48_000,
        "anthropic_medium_scoped_fix_model": TIER_MODEL,
        "anthropic_medium_scoped_fix_reasoning_effort": "high",
        "anthropic_medium_scoped_fix_max_tokens": 48_000,
    }
    values.update(overrides)
    return values


def test_an_over_ceiling_env_tier_starts_with_a_named_warning() -> None:
    """A bound over the declared ceiling is fixable, so the deployment starts and is told.

    The same pairing in a user-authored setup is refused at save; the split is deliberate
    (item 29): an env preset may predate the ceiling declaration, and the operational choice
    is a warning that names the variable and the ceiling, never a silent pass.
    """
    settings = anthropic_settings(
        model_max_output_tokens=f'{{"{TIER_MODEL}": 128000}}',
        **_medium_tier(anthropic_medium_coding_max_tokens=200_000),
    )

    warnings = settings.model_configuration_warnings
    assert any(
        "ANTHROPIC_MEDIUM_CODING_MAX_TOKENS" in warning
        and "128000" in warning
        and "exceeds" in warning
        for warning in warnings
    ), warnings


def test_the_181_vector_in_env_starts_with_the_exhaustion_warning() -> None:
    """The item's own vector: an env tier pairing xhigh with the bound flush at the ceiling.

    Expressible-but-named, not refused: the pairing is valid, and it is also the exact shape
    that exhausted AB-Feature-181 with zero headroom.
    """
    settings = anthropic_settings(
        model_max_output_tokens=f'{{"{TIER_MODEL}": 128000}}',
        **_medium_tier(
            anthropic_medium_coding_reasoning_effort="xhigh",
            anthropic_medium_coding_max_tokens=128_000,
        ),
    )

    assert any("AB-Feature-181" in warning for warning in settings.model_configuration_warnings), (
        settings.model_configuration_warnings
    )


def test_an_unsatisfiable_env_pairing_refuses_startup_naming_tier_variable_ceiling() -> None:
    """Effort `max` needs 64k and the ceiling is 32k: no bound value can ever work.

    This is the one ceiling finding startup refuses, and the message must carry what an
    operator edits: the tier, the effort variable actually read, and the ceiling.
    """
    with pytest.raises(ValidationError) as refused:
        anthropic_settings(
            model_max_output_tokens=f'{{"{TIER_MODEL}": 32000}}',
            **_medium_tier(
                anthropic_medium_coding_reasoning_effort="max",
                anthropic_medium_coding_max_tokens=32_000,
            ),
        )

    text = str(refused.value)
    assert "medium" in text
    assert "ANTHROPIC_MEDIUM_CODING_REASONING_EFFORT" in text
    assert "32000" in text


def test_a_clean_environment_starts_silently() -> None:
    """A configuration with no ceiling finding produces no warning at all.

    The declaration is pinned empty because the deployment `.env` this suite loads settings
    through declares real ceilings; this test decides its own configuration, the same way
    the floor tests pin the tier fields.
    """
    assert anthropic_settings(model_max_output_tokens="").model_configuration_warnings == ()


def test_an_undeclared_model_has_no_known_ceiling_and_no_finding() -> None:
    """A ceiling declared for some other model must not judge this tier's pairings."""
    settings = anthropic_settings(
        model_max_output_tokens='{"some-other-model": 16000}',
        **_medium_tier(anthropic_medium_coding_max_tokens=200_000),
    )

    assert settings.model_configuration_warnings == ()


def test_the_authoring_refusal_sentence_is_byte_identical() -> None:
    """The save-path over-ceiling refusal keeps its exact pre-74 sentence.

    The extraction into `judge_anthropic_ceiling` moved the sentence, and moving it is only
    safe if the authored path cannot tell: same rule, same words, byte for byte.
    """
    roles = parse_model_setup_roles(
        {
            "reasoning": {
                "platform": "anthropic",
                "model": "claude-opus-5",
                "reasoning_effort": "max",
                "max_tokens": 64_000,
            },
            "coding": {
                "platform": "anthropic",
                "model": "claude-sonnet-5",
                "reasoning_effort": "high",
                "max_tokens": 200_000,
            },
            "review": {
                "platform": "anthropic",
                "model": "claude-sonnet-5",
                "reasoning_effort": "high",
                "max_tokens": 48_000,
            },
            "scoped_fix": {
                "platform": "anthropic",
                "model": "claude-haiku-4-5",
                "max_tokens": 32_000,
            },
        }
    )

    with pytest.raises(ModelConfigurationError) as refused:
        validate_model_setup(roles, unsupported={}, ceilings={"claude-sonnet-5": 128_000})

    assert str(refused.value) == (
        "roles.coding.max_tokens=200000 exceeds claude-sonnet-5's declared output "
        "ceiling of 128000 (MODEL_MAX_OUTPUT_TOKENS); lower it to at most 128000"
    )
