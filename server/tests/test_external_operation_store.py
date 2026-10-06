"""`_safe_metadata` decides what a journal row is allowed to persist before it is written.

A credential-shaped key name alone used to be refused outright, with no look at the value
under it -- indistinguishable from a reviewer's own structured findings about
authentication-handling code it was asked to review. AB-Feature-182's backend reproduced this
sixteen times running: `run_reviewer` succeeded on every single call, then lost its result
here, because the review's own metadata named an ordinary field like
`credential_check_passed`, discarding a real, already-paid-for ~200 second review and forcing
the whole attempt to restart. These tests pin the fix: the key alone is still reason to look
closer, but only a value that itself passes the same high-confidence-secret check
`redact_source_credentials` already uses turns that look into a refusal.
"""

from __future__ import annotations

import pytest

from storage.external_operation_store import ExternalOperationError, _safe_metadata


def test_a_credential_shaped_key_with_an_ordinary_value_is_accepted() -> None:
    """AB-Feature-182's exact reproduction: review commentary, not a secret."""
    payload = {
        "review_artifact": {
            "verdict": "changes_requested",
            "metadata": {
                "credential_check_passed": False,
                "authorization_notes": (
                    "no credential leakage was found in the reviewed diff for this requirement"
                ),
            },
        }
    }

    accepted = _safe_metadata(payload)

    assert accepted == payload


def test_a_credential_shaped_key_with_a_secret_shaped_value_is_still_refused() -> None:
    """The regression this must never reopen: a real credential literal still blocks."""
    with pytest.raises(ExternalOperationError, match="credential-like values"):
        _safe_metadata({"api_key": "sk-proj-ACTUALSECRETVALUE1234567890"})


@pytest.mark.parametrize("value", [True, False, 0, 42, None])
def test_a_credential_shaped_key_with_a_non_string_value_is_accepted(value: object) -> None:
    """A bool, a count, or an absent value cannot itself be a credential literal."""
    accepted = _safe_metadata({"token_required": value})

    assert accepted == {"token_required": value}


def test_a_secret_shaped_value_nested_under_an_unrelated_key_is_still_caught() -> None:
    """Recursion into nested structures is unchanged: a real leak at depth still blocks."""
    payload = {
        "findings": [
            {"api_key": "sk-proj-ACTUALSECRETVALUE1234567890"},
        ]
    }

    with pytest.raises(ExternalOperationError, match="credential-like values"):
        _safe_metadata(payload)


def test_a_non_string_key_is_refused_regardless_of_the_value() -> None:
    """Unrelated to secret-shape: a non-string key cannot be JSON-persisted at all."""
    with pytest.raises(ExternalOperationError, match="credential-like values"):
        _safe_metadata({1: "a key that is not a string"})  # type: ignore[dict-item]
