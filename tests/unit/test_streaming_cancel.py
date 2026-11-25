"""Focused tests for prompt cancellation of an in-flight model stream."""

import asyncio
import time

import pytest
from langchain_core.messages import AIMessageChunk

from agent.streaming import _invoke_with_budget


class SlowStreamModel:
    def __init__(self):
        self.closed = False

    async def astream(self, messages):
        try:
            for index in range(100):
                yield AIMessageChunk(content=f"token-{index} ")
                await asyncio.sleep(0.01)
        finally:
            self.closed = True


@pytest.mark.asyncio
async def test_cancel_event_interrupts_stream_promptly_and_closes_it():
    model = SlowStreamModel()
    cancel_event = asyncio.Event()

    async def trigger():
        await asyncio.sleep(0.03)
        cancel_event.set()

    trigger_task = asyncio.create_task(trigger())
    started = time.monotonic()
    response = await _invoke_with_budget(
        model,
        [AIMessageChunk(content="hi")],
        deadline=time.monotonic() + 5,
        model_name="test",
        cancel_event=cancel_event,
    )
    elapsed = time.monotonic() - started
    await trigger_task

    assert elapsed < 0.5
    assert model.closed is True
    assert "interrupted by user" in str(response.content)
    assert response.additional_kwargs.get("interrupted") is True


@pytest.mark.asyncio
async def test_already_cancelled_event_returns_without_waiting():
    model = SlowStreamModel()
    cancel_event = asyncio.Event()
    cancel_event.set()

    started = time.monotonic()
    response = await _invoke_with_budget(
        model,
        [],
        deadline=time.monotonic() + 5,
        model_name="test",
        cancel_event=cancel_event,
    )

    assert time.monotonic() - started < 0.5
    assert response.additional_kwargs.get("interrupted") is True
    assert model.closed is True


@pytest.mark.asyncio
async def test_no_cancel_event_streams_normally():
    class ShortModel:
        async def astream(self, messages):
            yield AIMessageChunk(content="ok")

    response = await _invoke_with_budget(
        ShortModel(),
        [],
        deadline=time.monotonic() + 5,
        model_name="test",
    )

    assert "ok" in str(response.content)
    assert not response.additional_kwargs.get("interrupted")
