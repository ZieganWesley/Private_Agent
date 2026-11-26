"""Behavioural tests for the phase-aware CLI status indicators."""

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

from agent import status


class FakeStatus:
    def __init__(self, message, **kwargs):
        self.message = message
        self.kwargs = kwargs
        self.updated = []
        self.stopped = False

    def start(self):
        return self

    def update(self, message, **kwargs):
        self.message = message
        self.kwargs = kwargs
        self.updated.append((message, kwargs))

    def stop(self):
        self.stopped = True


class FakeConsole:
    def __init__(self, fail=False):
        self.fail = fail
        self.created = []

    def status(self, message, **kwargs):
        if self.fail:
            raise RuntimeError("console closed")
        status_object = FakeStatus(message, **kwargs)
        self.created.append(status_object)
        return status_object


def test_each_phase_has_a_distinct_spinner_and_colour():
    loading = status.phase_spinner(status.LOADING)
    thinking = status.phase_spinner(status.THINKING)
    processing = status.phase_spinner(status.PROCESSING)
    summarizing = status.phase_spinner(status.SUMMARIZING)

    assert loading[1] == "cyan"
    assert thinking[1] == "magenta"
    assert processing[1] == "cyan"
    assert summarizing[1] == "blue"
    assert thinking[0] != processing[0]
    assert summarizing[0] != processing[0]
    assert status.phase_spinner("unknown") == processing


def test_messages_describe_loading_and_processing():
    assert "test-model" in status.loading_message("test-model")
    assert "memory" in status.loading_message("test-model")
    assert status.waiting_message("m", local_model=True) == status.loading_message("m")
    assert status.waiting_message("m", local_model=False) == status.processing_message(
        "m"
    )
    assert "thinking" in status.thinking_message("m")


def test_summarizing_message_names_the_model_and_subject():
    assert status.summarizing_message("m") == "Summarizing the conversation with m"
    assert (
        status.summarizing_message("m", "tool results")
        == "Summarizing tool results with m"
    )


def test_reasoning_text_reads_kwargs_and_reasoning_blocks():
    from_kwargs = AIMessageChunk(
        content="", additional_kwargs={"reasoning_content": "step one"}
    )
    from_reasoning_field = AIMessage(
        content="", additional_kwargs={"reasoning": "step two"}
    )
    from_blocks = AIMessage(
        content=[
            {"type": "thinking", "text": "step three"},
            {"type": "text", "text": "visible"},
        ]
    )

    assert status.reasoning_text(from_kwargs) == "step one"
    assert status.reasoning_text(from_reasoning_field) == "step two"
    assert status.reasoning_text(from_blocks) == "step three"


def test_reasoning_text_prefers_blocks_over_mirrored_kwargs():
    message = AIMessage(
        content=[{"type": "thinking", "text": "from block"}],
        additional_kwargs={"reasoning_content": "from block"},
    )

    assert status.reasoning_text(message) == "from block"


def test_reasoning_text_ignores_missing_or_malformed_values():
    assert status.reasoning_text(None) == ""
    assert status.reasoning_text(AIMessage(content="visible")) == ""
    assert status.reasoning_text(AIMessage(content="", additional_kwargs={})) == ""
    assert (
        status.reasoning_text(
            AIMessage(content="", additional_kwargs={"reasoning_content": 42})
        )
        == ""
    )
    assert (
        status.reasoning_text(
            AIMessage(content=[{"type": "thinking", "text": None}, "plain"])
        )
        == ""
    )


def test_indicator_renders_the_colour_of_its_phase():
    console = FakeConsole()
    indicator = status.StatusIndicator(
        "Loading the model", phase=status.LOADING, console=console
    )

    indicator.start()

    created = console.created[0]
    assert created.message == "[bold cyan]Loading the model[/bold cyan]"
    assert created.kwargs == {"spinner": "dots12", "spinner_style": "cyan"}
    assert indicator.active


def test_indicator_update_switches_to_the_thinking_phase():
    console = FakeConsole()
    indicator = status.StatusIndicator("Waiting", console=console).start()

    indicator.update(phase=status.THINKING, message="m is thinking")

    created = console.created[0]
    assert created.message == "[bold magenta]m is thinking[/bold magenta]"
    assert created.kwargs == {"spinner": "dots2", "spinner_style": "magenta"}
    assert indicator.phase == status.THINKING


def test_indicator_start_and_stop_are_idempotent():
    console = FakeConsole()
    indicator = status.StatusIndicator("Waiting", console=console)

    indicator.start()
    indicator.start()
    indicator.stop()
    indicator.stop()

    assert len(console.created) == 1
    assert console.created[0].stopped
    assert not indicator.active


def test_indicator_works_as_a_context_manager():
    console = FakeConsole()

    with status.StatusIndicator("Waiting", console=console) as indicator:
        assert indicator.active

    assert console.created[0].stopped
    assert not indicator.active


def test_indicator_never_raises_when_the_console_fails():
    console = FakeConsole(fail=True)
    indicator = status.StatusIndicator("Waiting", console=console)

    indicator.start()
    indicator.update(phase=status.THINKING, message="thinking")
    indicator.stop()

    assert not indicator.active


def test_indicator_swallows_errors_from_a_broken_status_object():
    class BrokenStatus(FakeStatus):
        def stop(self):
            raise RuntimeError("already closed")

        def update(self, message, **kwargs):
            raise RuntimeError("already closed")

    class BrokenConsole(FakeConsole):
        def status(self, message, **kwargs):
            return BrokenStatus(message, **kwargs)

    indicator = status.StatusIndicator("Waiting", console=BrokenConsole()).start()

    indicator.update(phase=status.THINKING)
    indicator.stop()

    assert not indicator.active


def test_indicator_skips_an_empty_message():
    console = FakeConsole()
    indicator = status.StatusIndicator("", console=console)

    indicator.start()

    assert console.created == []
    assert not indicator.active


def test_context_bar_fills_with_the_share_of_the_window_in_use():
    empty = status.context_bar(0, 100, width=10)
    half = status.context_bar(50, 100, width=10)
    full = status.context_bar(100, 100, width=10)

    assert empty == "\u25b1" * 10 + " 0%"
    assert half == "\u25b0" * 5 + "\u25b1" * 5 + " 50%"
    assert full == "\u25b0" * 10 + " 100%"
    assert status.context_bar(25, 100, width=4) == "\u25b0\u25b1\u25b1\u25b1 25%"


def test_context_gauge_clamps_and_ignores_unusable_limits():
    assert status.context_bar(300, 100, width=10) == "\u25b0" * 10 + " 100%"
    assert status.context_bar(-5, 100, width=10).startswith("\u25b1" * 10)
    assert status.context_bar(10, 0) == ""
    assert status.context_bar(10, "unavailable") == ""
    assert status.context_percent(10, 0) == 0.0
    assert status.context_percent(5, 10) == 50.0


def test_context_style_warns_as_the_window_fills_up():
    assert status.context_style(10, 100) == "green"
    assert status.context_style(60, 100) == "yellow"
    assert status.context_style(80, 100) == "red"


def test_context_state_line_shows_the_gauge_and_budgets():
    line = status.context_state_line({"tokens": 3000, "max": 12000}, 2048)

    assert "[Context State]" in line
    assert "25%" in line
    assert "3,000 / 12,000 prompt tokens" in line
    assert "up to 2048 output tokens" in line
    assert "[green]" in line
    assert "[Context State]" in status.context_state_line({"tokens": 0, "max": 0}, 100)


def test_indicator_renders_the_context_gauge_when_supplied():
    console = FakeConsole()
    indicator = status.StatusIndicator(
        "Waiting", console=console, context_usage=(3000, 12000)
    )

    indicator.start()
    indicator.update(phase=status.THINKING, message="m is thinking")

    message = console.created[0].message
    assert message.startswith("[bold magenta]m is thinking[/bold magenta] (25%)")
    assert "25%" in message
    assert status.StatusIndicator("Waiting", console=FakeConsole())._render() == (
        "[bold cyan]Waiting[/bold cyan]"
    )


@pytest.mark.parametrize(
    "phase",
    [status.LOADING, status.THINKING, status.PROCESSING, status.SUMMARIZING],
)
def test_indicator_honours_every_known_phase(phase):
    console = FakeConsole()

    with status.StatusIndicator("Waiting", phase=phase, console=console):
        pass

    spinner, style = status.phase_spinner(phase)
    assert console.created[0].kwargs == {"spinner": spinner, "spinner_style": style}


def test_status_line_includes_model_context_and_permission():
    line = status.status_line("gpt-test", 2500, 10000, "full")
    assert "Model: gpt-test" in line
    assert "Context: 25.0%" in line
    assert "Permission: full" in line


def test_status_line_handles_missing_context():
    line = status.status_line("local", 0, 0, "auto")
    assert line == "Model: local | Context: 0.0% | Permission: auto | Background: 0"
