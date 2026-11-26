"""Unit tests for mid-turn steering state and input routing."""

import time

from agent.input_channel import InputChannel
from agent.turn_control import STEERING_PREFIX, TurnController


def test_begin_turn_resets_queue_and_flags():
    controller = TurnController()
    controller.queue_message("old")
    controller.request_stop()
    controller.begin_turn()

    assert controller.state == "generating"
    assert controller.queued_count == 0
    assert controller.stop_requested is False
    assert controller.cancel_event.is_set() is False


def test_queue_message_drains_in_order():
    controller = TurnController()
    controller.begin_turn()
    controller.queue_message("first")
    controller.queue_message("second")

    messages = controller.drain()

    assert [message.text for message in messages] == ["first", "second"]
    assert controller.queued_count == 0
    assert controller.drain() == []


def test_queue_is_bounded_by_message_count():
    controller = TurnController(max_messages=2)
    controller.begin_turn()

    assert controller.queue_message("one") is not None
    assert controller.queue_message("two") is not None
    assert controller.queue_message("three") is None
    assert controller.queued_count == 2


def test_queue_is_bounded_by_character_count():
    controller = TurnController(max_chars=5)
    controller.begin_turn()

    assert controller.queue_message("abc") is not None
    assert controller.queue_message("def") is None
    assert controller.queued_count == 1


def test_empty_steering_is_ignored():
    controller = TurnController()
    controller.begin_turn()

    assert controller.queue_message("   ") is None
    assert controller.queued_count == 0


def test_stop_wins_over_queue_limits_and_sets_event():
    controller = TurnController(max_messages=1)
    controller.begin_turn()
    controller.queue_message("one")

    assert controller.queue_message("stop now", mode="stop") is None
    assert controller.stop_requested is True
    assert controller.cancel_event.is_set() is True


def test_interrupt_sets_event_and_queues_message():
    controller = TurnController()
    controller.begin_turn()

    message = controller.queue_message("change direction", mode="interrupt")

    assert message is not None
    assert message.mode == "interrupt"
    assert controller.interrupt_requested is True
    assert controller.cancel_event.is_set() is True
    assert controller.queued_count == 1


def test_prompt_text_has_steering_prefix():
    controller = TurnController()
    controller.begin_turn()
    message = controller.queue_message("use the other file")

    assert message is not None
    assert message.prompt_text == f"{STEERING_PREFIX}use the other file"


def test_finish_turn_clears_cancel_flags():
    controller = TurnController()
    controller.begin_turn()
    controller.request_interrupt()

    controller.finish_turn()

    assert controller.state == "idle"
    assert controller.interrupt_requested is False
    assert controller.stop_requested is False


def test_input_channel_routes_stop():
    controller = TurnController()
    controller.begin_turn()
    channel = InputChannel(controller, is_tty=False, enabled=False)

    channel.submit("/stop")

    assert controller.stop_requested is True


def test_input_channel_routes_steer_as_interrupt():
    controller = TurnController()
    controller.begin_turn()
    channel = InputChannel(controller, is_tty=False, enabled=False)

    channel.submit("/steer focus on the failing test")

    messages = controller.drain()
    assert [message.text for message in messages] == ["focus on the failing test"]
    assert messages[0].mode == "interrupt"


def test_input_channel_queues_plain_text_and_slash_commands():
    controller = TurnController()
    controller.begin_turn()
    channel = InputChannel(controller, is_tty=False, enabled=False)

    channel.submit("plain instruction")
    channel.submit("/plan next")

    messages = controller.drain()
    assert [message.text for message in messages] == [
        "plain instruction",
        "/plan next",
    ]


def test_input_channel_is_inert_without_a_tty():
    controller = TurnController()
    channel = InputChannel(controller, is_tty=False)

    channel.start()

    assert channel.running is False
    channel.stop()


def test_input_channel_uses_a_stoppable_fd_reader(monkeypatch):
    import os

    from agent import input_channel

    read_fd, write_fd = os.pipe()

    class Stdin:
        def fileno(self):
            return read_fd

    monkeypatch.setattr(input_channel.sys, "stdin", Stdin())
    controller = TurnController()
    controller.begin_turn()
    channel = input_channel.InputChannel(controller, is_tty=True)
    try:
        channel.start()
        os.write(write_fd, b"queued from fd\n")
        for _ in range(100):
            if controller.queued_count:
                break
            time.sleep(0.005)
        channel.stop()

        assert channel.running is False
        assert controller.drain()[0].text == "queued from fd"
    finally:
        channel.stop()
        os.close(read_fd)
        os.close(write_fd)


def test_input_channel_stays_inert_for_a_non_fd_stdin(monkeypatch):
    from types import SimpleNamespace

    from agent import input_channel

    monkeypatch.setattr(
        input_channel.sys, "stdin", SimpleNamespace(isatty=lambda: True)
    )
    controller = TurnController()
    channel = input_channel.InputChannel(controller)

    channel.start()

    assert channel.running is False


def test_input_channel_reads_lines_in_a_thread():
    controller = TurnController()
    lines = iter(["queued from thread", ""])
    channel = InputChannel(
        controller,
        read_line=lambda: next(lines),
        is_tty=True,
    )

    channel.start()
    for _ in range(100):
        if not channel.running:
            break
        time.sleep(0.005)
    channel.stop()

    assert controller.queued_count == 1
    assert controller.drain()[0].text == "queued from thread"
