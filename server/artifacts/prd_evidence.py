"""The optional evidence a PRD may carry, and why an empty one is not carried at all.

Two 89- items each gave the PRD a list of evidence somebody may attach: `design_references`,
the Figma frames a request cites, and `attachments`, the images somebody pasted. Both are
optional, both default to an empty list, and both made the same promise -- a feature that
attached nothing must behave in every respect as it did before the field existed.

That promise is about bytes, not about intent. `{{ prd }}` in the product-manager prompt is
the whole PRD artifact serialized, and `_feature_fingerprint` hashes the whole submission, so
a key added to either reaches every feature's first model call and changes every idempotency
key. `"design_references": []` in front of a model is an instruction to think about designs on
a feature that has none, and `"attachments": []` is the same instruction about pictures.

So both keys are omitted when empty, and both are kept when they are not, at serialization
rather than by each caller -- so the prompt, the API response and the persisted row all agree
without three of them remembering to. One function rather than two because the rule is one
rule; the next optional evidence list joins the tuple below and inherits it.

This module is a leaf and imports nothing from this repository, so `api.schemas` and
`artifacts.schemas` can both hold the rule without either importing the other.
"""

from __future__ import annotations

from typing import Any

# Every PRD field that is optional evidence: absent by default, present only because somebody
# supplied it. The order is the order the fields were added, which is also the order they are
# declared on both schemas.
EMPTY_WHEN_UNSUPPLIED = ("attachments", "design_references")


def without_empty_prd_evidence(payload: dict[str, Any]) -> dict[str, Any]:
    """Drop each optional evidence key from a serialized PRD when nothing was supplied for it.

    Mutates and returns the mapping it is given, which is always a fresh dict from a
    `model_serializer(mode="wrap")` handler -- never a caller's own.
    """
    for field in EMPTY_WHEN_UNSUPPLIED:
        if not payload.get(field):
            payload.pop(field, None)
    return payload


__all__ = ["EMPTY_WHEN_UNSUPPLIED", "without_empty_prd_evidence"]
