"""The bordered input box.

Built as a small prompt-toolkit `Application` rather than a `PromptSession`,
because the box has to stay closed while the user types and grow with the
text. The application is inline (no alternate screen) and erases itself on
accept, so ordinary scrollback stays intact and the handler is free to
re-print the prompt however it likes.
"""

from abc import ABC, abstractmethod
from typing import Iterable, Optional, Sequence, Tuple

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import (
    Float,
    FloatContainer,
    HSplit,
    VSplit,
    Window,
)
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.styles import BaseStyle

from src.handler.cli.commands import COMMAND_PREFIX


class PromptReader(ABC):
    """Input side of the terminal handler."""

    @abstractmethod
    async def read(self) -> Optional[str]:
        """Return the submitted line, or `None` when the user wants to leave."""
        raise NotImplementedError()


class _CommandCompleter(Completer):
    """Suggests slash commands, and only while the line starts with one."""

    def __init__(self, commands: Sequence[Tuple[str, str]]) -> None:
        self._commands = tuple(commands)

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        typed = document.text_before_cursor
        if not typed.startswith(COMMAND_PREFIX) or "\n" in typed:
            return
        for name, summary in self._commands:
            if name.startswith(typed):
                yield Completion(
                    name,
                    start_position=-len(typed),
                    display=name,
                    display_meta=summary,
                )


def _border(char: str) -> Window:
    return Window(width=1, char=char, style="class:frame.border")


def _rule(left: str, right: str) -> VSplit:
    """One horizontal edge of the box, stretched to the terminal width."""
    return VSplit(
        [
            Window(width=1, height=1, char=left, style="class:frame.border"),
            Window(height=1, char="─", style="class:frame.border"),
            Window(width=1, height=1, char=right, style="class:frame.border"),
        ]
    )


class TerminalPromptReader(PromptReader):
    """Rounded, auto-growing input box with history and command completion."""

    def __init__(
        self,
        *,
        marker: str,
        history_path: str,
        style: BaseStyle,
        commands: Sequence[Tuple[str, str]],
        hint: str,
    ) -> None:
        self._marker = marker
        self._continuation = " " * len(marker)
        self._hint = hint

        self._buffer = Buffer(
            multiline=True,
            history=FileHistory(history_path),
            completer=_CommandCompleter(commands),
            complete_while_typing=True,
        )
        self._application = self._build_application(style)

    async def read(self) -> Optional[str]:
        self._buffer.reset()
        return await self._application.run_async()

    def _line_prefix(self, line_number: int, wrap_count: int) -> StyleAndTextTuples:
        """Caret on the first line, blank gutter of the same width below it."""
        if line_number == 0 and wrap_count == 0:
            return [("class:prompt.marker", self._marker)]
        return [("class:prompt.continuation", self._continuation)]

    def _build_application(self, style: BaseStyle) -> Application:
        input_window = Window(
            content=BufferControl(buffer=self._buffer),
            get_line_prefix=self._line_prefix,
            wrap_lines=True,
            dont_extend_height=True,
        )

        box = HSplit(
            [
                _rule("╭", "╮"),
                VSplit(
                    [
                        _border("│"),
                        Window(width=1, char=" "),
                        input_window,
                        Window(width=1, char=" "),
                        _border("│"),
                    ]
                ),
                _rule("╰", "╯"),
                Window(
                    content=FormattedTextControl(
                        lambda: [("class:hint", f" {self._hint}")]
                    ),
                    height=1,
                ),
            ]
        )

        root = FloatContainer(
            content=box,
            floats=[
                Float(
                    xcursor=True,
                    ycursor=True,
                    content=CompletionsMenu(max_height=8, scroll_offset=1),
                )
            ],
        )

        return Application(
            layout=Layout(root, focused_element=input_window),
            key_bindings=self._build_key_bindings(),
            style=style,
            full_screen=False,
            erase_when_done=True,
            mouse_support=False,
        )

    def _build_key_bindings(self) -> KeyBindings:
        bindings = KeyBindings()
        buffer = self._buffer

        @bindings.add("enter")
        def _submit(event: KeyPressEvent) -> None:
            """Send the prompt, unless a completion is being picked."""
            state = buffer.complete_state
            if state is not None and state.current_completion is not None:
                buffer.apply_completion(state.current_completion)
                return
            if not buffer.text.strip():
                return
            buffer.append_to_history()
            event.app.exit(result=buffer.text)

        @bindings.add("escape", "enter")
        @bindings.add("c-j")
        def _newline(event: KeyPressEvent) -> None:
            """Alt+Enter / Ctrl+J for a multi-line prompt."""
            buffer.insert_text("\n")

        @bindings.add("c-c")
        def _abandon(event: KeyPressEvent) -> None:
            """Throw the draft away without leaving the session."""
            buffer.reset()

        @bindings.add("c-d")
        def _leave(event: KeyPressEvent) -> None:
            """Leave, but only from an empty box, so it cannot fire by accident."""
            if buffer.text:
                return
            event.app.exit(result=None)

        @bindings.add("tab")
        def _complete(event: KeyPressEvent) -> None:
            """Open the command menu, or step through it."""
            if buffer.complete_state is None:
                buffer.start_completion(select_first=True)
            else:
                buffer.complete_next()

        @bindings.add("s-tab")
        def _complete_back(event: KeyPressEvent) -> None:
            if buffer.complete_state is not None:
                buffer.complete_previous()

        @bindings.add("up")
        def _older(event: KeyPressEvent) -> None:
            """Walk the open menu, then the current prompt, then history."""
            if buffer.complete_state is not None:
                buffer.complete_previous()
            elif buffer.document.cursor_position_row == 0:
                buffer.history_backward()
            else:
                buffer.cursor_up()

        @bindings.add("down")
        def _newer(event: KeyPressEvent) -> None:
            if buffer.complete_state is not None:
                buffer.complete_next()
            elif buffer.document.cursor_position_row == buffer.document.line_count - 1:
                buffer.history_forward()
            else:
                buffer.cursor_down()

        return bindings
