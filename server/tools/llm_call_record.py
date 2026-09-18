"""The model facts of the most recent LLM response, kept task-locally for the journal.

The pre-coding planning calls -- the product-manager draft, requirement reconciliation,
clarification grounding, reconnaissance, the planner -- are journaled as operation rows, but
two of them return values that carry no model metadata at all (a ``RequirementReconciliation``,
a list of questions), so the row could not say which model answered it and every one of those
calls rendered as "Model not recorded". The facts provably exist in exactly one place: the
adapter, at the moment it builds the ``LLMResponse``. This module is that moment made readable
from the journaling seam.

The mechanics matter and are deliberate:

* **A ``ContextVar``, not module state.** The journaled action runs inside its own asyncio
  task (the heartbeat wrapper creates it), and concurrent reconnaissance calls run in sibling
  tasks. A context variable set inside one task is invisible to the others, so two calls in
  flight can never read each other's model.

* **Reset before the call, read after it, in the same task.** The journaling wrapper resets
  the record, awaits the agent call, then reads. A call that never reached a provider -- a
  deterministic composition, a mock run -- therefore reads nothing, and the row honestly says
  no model was recorded rather than inheriting a neighbour's.

* **Last response wins.** A call that repairs a rejected response makes two ``respond``
  round-trips; the repair response is the one whose output became the artifact, and it is the
  one the row names.

The stored shape is the artifact vocabulary: the same ``{"execution": {...}}`` block
``execution_metadata`` stamps on artifacts, minus ``agent_type`` (the row's ``logical_step``
already answers that), so ``execution_records._model_facts`` reads a journal payload and an
artifact with one reader.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

# The key the execution-records reader already consumes. Spelled here rather than imported:
# this module must stay a leaf (the adapters import it), and the reader's own constant lives
# above the layers that may import adapters.
EXECUTION_BLOCK_KEY = "execution"

_LAST_CALL: ContextVar[dict[str, Any] | None] = ContextVar("last_llm_call", default=None)


def record_llm_call(
    *,
    model: str,
    provider: str | None = None,
    reasoning_effort: str | None = None,
    model_role: str | None = None,
    model_variable: str | None = None,
    routing_reason: str | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    stream_reissues: int | None = None,
) -> None:
    """Note what one LLM response was produced by, for the journal row wrapping this task.

    Called by the adapters at the moment the response exists -- never reconstructed from
    configuration afterwards. Carries no credential, no base URL and no prompt; the usage
    counts keep the ``usage_`` spelling because persistence screens metadata keys for
    credential-shaped markers and "token" is one.
    """
    block: dict[str, Any] = {"model": model}
    if provider:
        block["provider"] = provider
    if reasoning_effort:
        block["reasoning_effort"] = reasoning_effort
    if model_role:
        block["model_role"] = model_role
    if model_variable:
        block["model_variable"] = model_variable
    if routing_reason:
        block["routing_reason"] = routing_reason
    if input_tokens is not None:
        block["usage_input"] = input_tokens
    if output_tokens is not None:
        block["usage_output"] = output_tokens
    if stream_reissues is not None:
        # Recorded even when zero, unlike every field above it. Those are facts about what
        # answered; this one is a measurement, and a measurement that is only stored when it
        # is interesting cannot tell "the budget was never close" from "nobody looked".
        block["stream_reissues"] = stream_reissues
    _LAST_CALL.set(block)


def reset_llm_call_record() -> None:
    """Clear the record for one journaled call, so no call can inherit another's model."""
    _LAST_CALL.set(None)


def last_llm_call_payload() -> dict[str, Any] | None:
    """Return the journal payload for the most recent response in this task, or nothing.

    Shaped as ``{"execution": {...}}`` so the operation's ``result_payload`` speaks the same
    vocabulary an artifact's metadata does, and the read model needs exactly one reader.
    """
    block = _LAST_CALL.get()
    if block is None:
        return None
    return {EXECUTION_BLOCK_KEY: dict(block)}
