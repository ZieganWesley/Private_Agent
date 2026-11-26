"""Measure event-loop stalls of runtime turn persistence / task reads.

Run from the repo root with PYTHONPATH=src. Prints the worst gap between
1 ms ticks while the DB call is in flight, using a fake memory whose calls
sleep like a slow SQLite query.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from types import SimpleNamespace

import agent.runtime as runtime


class SlowMemory:
    def __init__(self, delay: float = 0.05) -> None:
        self.delay = delay
        self.threads: list[str] = []

    def save_message(self, *args, **kwargs):
        import threading

        self.threads.append(threading.current_thread().name)
        time.sleep(self.delay)
        return None

    def get_task(self, *args, **kwargs):
        import threading

        self.threads.append(threading.current_thread().name)
        time.sleep(self.delay)
        return {"task_id": "t", "goal": "g"}


class Transcript:
    def add_message(self, *args, **kwargs):
        return None

    def stream_chunk(self, *args, **kwargs):
        return None


async def ticker(stop: asyncio.Event, gaps: list[float]) -> None:
    last = time.perf_counter()
    while not stop.is_set():
        await asyncio.sleep(0.001)
        now = time.perf_counter()
        gaps.append(now - last)
        last = now


async def measure(coro_factory, label: str) -> None:
    memory = SlowMemory()
    stop = asyncio.Event()
    gaps: list[float] = []
    task = asyncio.create_task(ticker(stop, gaps))
    await asyncio.sleep(0.005)
    started = time.perf_counter()
    await coro_factory(memory)
    elapsed = time.perf_counter() - started
    stop.set()
    await task
    worst = max(gaps) * 1000
    print(
        f"{label}: wall={elapsed * 1000:.1f}ms  worst_tick_gap={worst:.1f}ms  "
        f"thread={memory.threads[0] if memory.threads else 'n/a'}"
    )


async def main() -> None:
    async def persist(memory):
        call = runtime._store_turn_output(
            response=SimpleNamespace(content="hello"),
            session_ui=SimpleNamespace(print_markup=lambda *a, **k: None),
            chat_transcript=Transcript(),
            retrieved_citations=[],
            active_code_workspace=None,
            cleanup_code_task=lambda **kwargs: None,
            memory=memory,
            session_id="s",
            user_input="q",
            turn_id="t1",
            chat_history=[],
        )
        if inspect.isawaitable(call):
            await call

    async def read_task(memory):
        loader = getattr(runtime, "_load_task", None)
        if loader is None:
            memory.get_task("t")  # old inline pattern
            return
        await loader(memory, "t")

    await measure(persist, "_store_turn_output (cold)")
    await measure(persist, "_store_turn_output (warm)")
    await measure(read_task, "_load_task / get_task (cold)")
    await measure(read_task, "_load_task / get_task (warm)")


asyncio.run(main())
