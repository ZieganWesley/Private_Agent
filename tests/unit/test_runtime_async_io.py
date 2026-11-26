"""Runtime persistence must not run SQLite/IO on the event loop."""

from __future__ import annotations

import asyncio
import inspect
import threading
import time
from types import SimpleNamespace

from agent import runtime


class _Transcript:
    def add_message(self, *args, **kwargs) -> None:
        return None

    def stream_chunk(self, *args, **kwargs) -> None:
        return None


class _SlowMemory:
    """Fake memory whose calls sleep like a slow SQLite query."""

    def __init__(self, delay: float = 0.12) -> None:
        self.delay = delay
        self.calls: list[tuple[str, str]] = []

    def save_message(self, session_id, role, content, *, turn_id=None) -> None:
        self.calls.append((role, content))
        self.assert_worker_thread()
        time.sleep(self.delay)

    def get_task(self, task_id):
        self.assert_worker_thread()
        time.sleep(self.delay)
        return {"task_id": task_id}

    @staticmethod
    def assert_worker_thread() -> None:
        assert threading.current_thread() is not threading.main_thread(), (
            "DB call ran on the event loop thread"
        )


def _store_output(memory: _SlowMemory):
    return runtime._store_turn_output(
        response=SimpleNamespace(content="hello"),
        session_ui=SimpleNamespace(print_markup=lambda *args, **kwargs: None),
        chat_transcript=_Transcript(),
        retrieved_citations=[],
        active_code_workspace=None,
        cleanup_code_task=lambda **kwargs: None,
        memory=memory,
        session_id="session",
        user_input="question",
        turn_id="turn",
        chat_history=[],
    )


def test_store_turn_output_persists_off_the_event_loop() -> None:
    assert inspect.iscoroutinefunction(runtime._store_turn_output)
    memory = _SlowMemory()

    async def scenario() -> list[float]:
        gaps: list[float] = []
        stop = asyncio.Event()

        async def ticker() -> None:
            last = time.perf_counter()
            while not stop.is_set():
                await asyncio.sleep(0.002)
                now = time.perf_counter()
                gaps.append(now - last)
                last = now

        ticking = asyncio.create_task(ticker())
        await asyncio.sleep(0.01)
        content = await _store_output(memory)
        stop.set()
        await ticking
        assert content == "hello"
        return gaps

    gaps = asyncio.run(scenario())

    assert memory.calls == [("human", "question"), ("ai", "hello")]
    assert max(gaps) < 0.1, f"event loop stalled for {max(gaps):.3f}s"


def test_load_task_reads_off_the_event_loop() -> None:
    memory = _SlowMemory(delay=0)

    task = asyncio.run(runtime._load_task(memory, "task-1"))
    assert task == {"task_id": "task-1"}

    assert asyncio.run(runtime._load_task(memory, None)) is None
    assert asyncio.run(runtime._load_task(memory, "")) is None
