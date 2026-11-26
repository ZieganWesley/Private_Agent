"""Tool for asking the user a clarifying question during planning."""

import sys

from langchain_core.tools import tool

from .schemas import AskUserInput


@tool(args_schema=AskUserInput)
def ask_user(question: str, options: list[str] | None = None) -> str:
    """Ask the user a clarifying question and return their answer."""
    from ui.console import console
    from ui.markup import literal

    # The model writes the question and options: escape them so their brackets
    # are shown to the user instead of restyling the prompt.
    console.print(f"[info][Question][/info] {literal(question)}")
    if options:
        for index, option in enumerate(options, 1):
            console.print(f"  {index}. {literal(option)}")
    if not sys.stdin.isatty():
        return (
            "Non-interactive session: the question was shown above; "
            "wait for the user's next message before continuing."
        )
    try:
        answer = console.input("Answer: ").strip()
    except EOFError, KeyboardInterrupt:
        return "The user did not answer the question."
    if not answer:
        return "The user did not provide an answer."
    return f"User answered: {answer}"
