import threading

from private_agent import events


def test_emit_calls_subscribers():
    bus = events.reset_event_bus()
    received = []

    def handler(e):
        received.append(e)

    bus.subscribe(events.Event, handler)
    bus.emit(events.ToolStarted(call_id="c1", tool_name="read"))
    assert len(received) == 1
    assert isinstance(received[0], events.ToolStarted)


def test_subscriber_specific_type():
    bus = events.reset_event_bus()
    tool_events = []

    def handler(e):
        tool_events.append(e)

    bus.subscribe(events.ToolStarted, handler)
    bus.emit(events.Event())
    bus.emit(events.ToolStarted(call_id="c1", tool_name="read"))
    assert len(tool_events) == 1
    assert tool_events[0].tool_name == "read"


def test_handler_error_isolated():
    bus = events.reset_event_bus()
    good = []

    def bad(e):
        raise RuntimeError("boom")

    def ok(e):
        good.append(e)

    bus.subscribe(events.Event, bad)
    bus.subscribe(events.Event, ok)
    bus.emit(events.Event())
    assert len(good) == 1


def test_unsubscribe():
    bus = events.reset_event_bus()
    received = []

    def handler(e):
        received.append(e)

    bus.subscribe(events.Event, handler)
    bus.unsubscribe(events.Event, handler)
    bus.emit(events.Event())
    assert received == []


def test_thread_safety():
    bus = events.reset_event_bus()
    counter = {"n": 0}
    lock = threading.Lock()

    def handler(e):
        with lock:
            counter["n"] += 1

    bus.subscribe(events.Event, handler)
    threads = [
        threading.Thread(target=lambda: bus.emit(events.Event())) for _ in range(50)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert counter["n"] == 50


def test_reset_replaces_bus():
    b1 = events.get_event_bus()
    b2 = events.reset_event_bus()
    assert b1 is not b2
