"""The task-local record of what one LLM call ran on.

The journal rows for pre-coding planning calls name their model by reading this record inside
the journaled action, right after the agent call returns. Three properties make that honest,
and each gets pinned here: the record is task-local (concurrent calls cannot read each
other's), the last response wins (a repair's response is the one whose output became the
artifact), and reset clears it (a call that never reached a provider reads nothing rather
than inheriting a neighbour's model).
"""

from __future__ import annotations

import asyncio

import pytest

from tools.llm_call_record import last_llm_call_payload, record_llm_call, reset_llm_call_record

pytestmark = pytest.mark.asyncio


async def test_the_last_recorded_call_wins_and_reset_clears_it() -> None:
    """A repair call's response replaces the rejected one's; a reset leaves nothing."""
    reset_llm_call_record()
    assert last_llm_call_payload() is None

    record_llm_call(model="first-model", provider="openai")
    record_llm_call(
        model="repair-model",
        provider="openai",
        reasoning_effort="high",
        model_role="reasoning",
        model_variable="OPENAI_REASONING_MODEL",
        routing_reason="Planning and high-level analysis use the configured reasoning role.",
        input_tokens=11,
        output_tokens=7,
    )

    payload = last_llm_call_payload()
    assert payload == {
        "execution": {
            "model": "repair-model",
            "provider": "openai",
            "reasoning_effort": "high",
            "model_role": "reasoning",
            "model_variable": "OPENAI_REASONING_MODEL",
            "routing_reason": (
                "Planning and high-level analysis use the configured reasoning role."
            ),
            "usage_input": 11,
            "usage_output": 7,
        }
    }

    reset_llm_call_record()
    assert last_llm_call_payload() is None


async def test_empty_optionals_stay_off_the_record() -> None:
    """A blank provider or absent effort is not an answer and is not written as one."""
    reset_llm_call_record()
    record_llm_call(model="bare-model", provider=None, reasoning_effort=None)
    assert last_llm_call_payload() == {"execution": {"model": "bare-model"}}


async def test_concurrent_calls_cannot_read_each_others_model() -> None:
    """Two journaled actions in sibling tasks each see only their own response.

    This is the shape the executor actually runs: each journaled action is its own asyncio
    task, the way `_run_with_heartbeat` creates one, and reconnaissance runs one per
    repository concurrently. A module-level record would let the slower call report the
    faster call's model; the ContextVar makes that impossible, and this test is the fence.
    """

    async def one_call(model: str, delay: float) -> dict[str, object] | None:
        reset_llm_call_record()
        await asyncio.sleep(delay)
        record_llm_call(model=model)
        await asyncio.sleep(delay)
        return last_llm_call_payload()

    first, second = await asyncio.gather(
        asyncio.create_task(one_call("model-a", 0.01)),
        asyncio.create_task(one_call("model-b", 0.005)),
    )
    assert first == {"execution": {"model": "model-a"}}
    assert second == {"execution": {"model": "model-b"}}
