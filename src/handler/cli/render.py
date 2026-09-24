"""Everything CORE prints between two input boxes.

The handler never touches the console directly: it describes *what* happened
and the renderer decides how it looks. That keeps the loop readable and makes
the terminal presentation replaceable on its own.
"""

import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Mapping, Optional, Sequence
from uuid import UUID

from rich.console import Console, RenderableType
from rich.live import Live
from rich.markdown import Markdown
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from src.entities.event import AgentEvent, TextChunk, ToolCall
from src.entities.message import Message
from src.entities.session import SessionDigest
from src.handler.cli.commands import Command
from src.handler.cli.logo import BADGE, LOGO_LINES, LOGO_WIDTH, NAME, TAGLINE
from src.handler.cli.theme import CODE_THEME
from src.utils.enums.message_role import MessageRole

AGENT_MARKER = "●"
USER_MARKER = "›"
NOTICE_MARKER = "·"
ERROR_MARKER = "✗"
TOOL_MARKER = "▸"
RESULT_MARKER = "└"

ELLIPSIS = "…"
GUTTER_WIDTH = 4
"""Columns a marked row spends on its gutter: the marker plus its padding."""

CURRENT_MARKER = "●"
MODEL_WIDTH = 16
"""Columns a model tag may take in the session list before it is clipped."""
TIMESTAMP_FORMAT = "%d %b %H:%M"
PREVIEW_LIMIT = 60
"""Characters of a session's first prompt kept in the session list."""


class Renderer(ABC):
    """Output side of the terminal handler."""

    @abstractmethod
    def banner(self, workspace: str, session: Optional[str] = None) -> None:
        """Greet the user, naming the conversation they are in when it has one."""
        raise NotImplementedError()

    @abstractmethod
    def prompt_echo(self, text: str) -> None:
        """Re-print the submitted prompt, which the input box erased."""
        raise NotImplementedError()

    @abstractmethod
    async def stream_turn(self, events: AsyncIterator[AgentEvent]) -> None:
        """Show one agent turn as it unfolds."""
        raise NotImplementedError()

    @abstractmethod
    def sessions(
        self, digests: Sequence[SessionDigest], current: Optional[UUID]
    ) -> None:
        """List the sessions that can be reopened, numbered as `/resume` takes them."""
        raise NotImplementedError()

    @abstractmethod
    def transcript(self, messages: Sequence[Message]) -> None:
        """Replay a stored conversation into the scrollback."""
        raise NotImplementedError()

    @abstractmethod
    def notice(self, text: str) -> None:
        """Show an unremarkable status message."""
        raise NotImplementedError()

    @abstractmethod
    def error(self, text: str) -> None:
        """Show a failure without ending the session."""
        raise NotImplementedError()

    @abstractmethod
    def help(self) -> None:
        """List the slash commands."""
        raise NotImplementedError()

    @abstractmethod
    def clear(self) -> None:
        """Wipe the screen."""
        raise NotImplementedError()

    @abstractmethod
    def farewell(self) -> None:
        """Sign off, once, at session end."""
        raise NotImplementedError()


class _Status:
    """The animated line shown while the agent has nothing to display yet.

    Rendered fresh on every `Live` refresh rather than updated by the caller,
    so the frames keep turning and the elapsed count keeps rising even while
    the agent is busy and sending nothing.
    """

    def __init__(self, *, label: str, hint: str, frames: str, style: str) -> None:
        self._label = label
        self._hint = hint
        self._spinner = Spinner(frames, style=style)
        self._started = time.monotonic()

    def __rich__(self) -> RenderableType:
        now = time.monotonic()
        caption = Text(
            f"{self._label} ({int(now - self._started)}s · {self._hint})",
            style="core.muted",
        )
        row = Table.grid(padding=(0, 1))
        row.add_row(self._spinner.render(now), caption)
        return row


class TerminalRenderer(Renderer):
    """Rich-backed renderer."""

    def __init__(
        self,
        console: Console,
        commands: Sequence[Command],
        thinking_label: str,
        thinking_hint: str,
        spinner_frames: str,
        spinner_style: str,
        refresh_rate: int,
        argument_limit: int,
        result_limit: int,
    ) -> None:
        self._console = console
        self._commands = tuple(commands)
        self._thinking_label = thinking_label
        self._thinking_hint = thinking_hint
        self._spinner_frames = spinner_frames
        self._spinner_style = spinner_style
        self._refresh_rate = refresh_rate
        self._argument_limit = argument_limit
        self._result_limit = result_limit

    def banner(self, workspace: str, session: Optional[str] = None) -> None:
        console = self._console
        console.print()
        if console.width >= LOGO_WIDTH + 2:
            for line in LOGO_LINES:
                console.print(Text(line, style="core.accent"))
        else:
            console.print(Text(BADGE, style="bold core.accent"))
        console.print()
        console.print(Text(f"  {NAME} · {TAGLINE}", style="core.muted"))
        console.print(Text(f"  workspace   {workspace}", style="core.faint"))
        if session is not None:
            console.print(
                Text(
                    f"  session     {self._summarize(session, PREVIEW_LIMIT)}",
                    style="core.faint",
                )
            )
        console.print(Text("  /help for commands · ctrl+d to exit", style="core.faint"))
        console.print()

    def prompt_echo(self, text: str) -> None:
        self._console.print(
            self._marked(USER_MARKER, "core.faint", Text(text, style="core.text"))
        )

    async def stream_turn(self, events: AsyncIterator[AgentEvent]) -> None:
        """Draw a turn while it happens.

        One `Live` region for the whole turn, holding only what is still
        changing: the status line, or the paragraph currently being written.
        Everything finished is printed through the console instead, which
        pushes it above that region and into the scrollback for good. So the
        transcript accumulates in the order the work happened, and the last
        frame is erased on the way out rather than left behind as a duplicate.
        """
        status = _Status(
            label=self._thinking_label,
            hint=self._thinking_hint,
            frames=self._spinner_frames,
            style=self._spinner_style,
        )
        pending = ""

        self._console.print()
        with Live(
            status,
            console=self._console,
            refresh_per_second=self._refresh_rate,
            transient=True,
            vertical_overflow="visible",
        ) as live:
            async for event in events:
                if isinstance(event, TextChunk):
                    pending += event.body
                    live.update(self._answer(pending))
                    continue

                pending = self._commit(pending)
                if isinstance(event, ToolCall):
                    self._console.print(self._tool_call(event.name, event.arguments))
                else:
                    self._console.print(self._tool_result(event.body))
                live.update(status)

            self._commit(pending)

        self._console.print()

    def _answer(self, body: str) -> RenderableType:
        return self._marked(
            AGENT_MARKER,
            "core.accent",
            Markdown(body, code_theme=CODE_THEME),
        )

    def _commit(self, pending: str) -> str:
        """Move a finished paragraph out of the live region into the scrollback."""
        if pending.strip():
            self._console.print(self._answer(pending))
        return ""

    def _tool_call(self, name: str, arguments: Mapping[str, Any]) -> RenderableType:
        return self._one_line(
            TOOL_MARKER,
            "core.tool",
            Text.assemble(
                (name, "core.tool.name"),
                (f"({self._arguments(arguments)})", "core.tool"),
            ),
        )

    def _tool_result(self, body: str) -> RenderableType:
        return self._one_line(
            RESULT_MARKER,
            "core.tool.result",
            Text(
                self._summarize(body, self._result_limit),
                style="core.tool.result",
            ),
        )

    def _one_line(self, marker: str, marker_style: str, body: Text) -> Table:
        """A gutter row that is clipped rather than wrapped.

        Tool activity is a log of what happened, not content to read: a call
        that spilled onto three lines would push the answer off the screen and
        bury the thing the user actually asked for.
        """
        body.truncate(
            max(self._console.width - GUTTER_WIDTH, GUTTER_WIDTH),
            overflow="ellipsis",
        )
        row = Table.grid(padding=(0, 1))
        row.add_column(width=1, no_wrap=True)
        row.add_column(no_wrap=True, overflow="ellipsis")
        row.add_row(Text(marker, style=marker_style), body)
        return row

    def _arguments(self, arguments: Mapping[str, Any]) -> str:
        """Render call arguments on one line, short enough to stay readable.

        A tool that writes a file carries the whole file in its arguments, so
        every value is clipped: the point is to show which call was made, not
        to reproduce its payload.
        """
        return ", ".join(
            f"{name}={self._summarize(str(value), self._argument_limit)}"
            for name, value in arguments.items()
        )

    def _summarize(self, body: str, limit: int) -> str:
        """Collapse a value to a single clipped line."""
        collapsed = " ".join(body.split())
        if len(collapsed) <= limit:
            return collapsed
        return collapsed[:limit] + ELLIPSIS

    def sessions(
        self, digests: Sequence[SessionDigest], current: Optional[UUID]
    ) -> None:
        """Numbered by position, because that is what `/resume` is given.

        The number is not stored anywhere: it means "the nth most recently used
        session" and is resolved against a freshly read list, so what the user
        reads here and what the handler reopens cannot drift apart.

        A real table rather than a grid, because only a table can say which
        column may shrink. Everything but the preview is fixed width: at eighty
        columns a grid divides the room evenly and eats the number the user is
        supposed to type.
        """
        if not digests:
            self.notice("No saved sessions yet. Ask something to start one.")
            return

        table = Table(
            box=None,
            show_header=False,
            pad_edge=False,
            padding=(0, 1),
            expand=True,
        )
        table.add_column(width=1, no_wrap=True)
        table.add_column(style="core.command", justify="right", no_wrap=True)
        table.add_column(style="core.muted", no_wrap=True)
        table.add_column(style="core.faint", justify="right", no_wrap=True)
        table.add_column(style="core.faint", no_wrap=True, max_width=MODEL_WIDTH)
        table.add_column(style="core.text", no_wrap=True, overflow="ellipsis", ratio=1)

        for position, digest in enumerate(digests, start=1):
            here = digest.session.id is not None and digest.session.id == current
            table.add_row(
                Text(CURRENT_MARKER if here else " ", style="core.accent"),
                str(position),
                self._when(digest.activity),
                f"{digest.message_count} msg",
                digest.session.model,
                " ".join(digest.preview.split()) or "—",
            )

        self._console.print()
        self._console.print(table)
        self._console.print(
            Text("  /resume <number> to reopen one", style="core.faint")
        )
        self._console.print()

    def transcript(self, messages: Sequence[Message]) -> None:
        """Print a stored conversation the way it was printed when it happened.

        Same renderings as the live stream — the assistant's Markdown, the
        clipped tool lines — so a resumed session reads as a continuation of
        the screen rather than as a database dump.
        """
        if not messages:
            self.notice("This session has no messages yet.")
            return

        console = self._console
        for message in messages:
            if message.role is MessageRole.USER:
                console.print()
                self.prompt_echo(message.body)
                console.print()
            elif message.role is MessageRole.ASSISTANT:
                console.print(self._answer(message.body))
            elif message.role is MessageRole.TOOL_CALL:
                console.print(
                    self._tool_call(
                        message.tool_name or "", message.tool_arguments or {}
                    )
                )
            else:
                console.print(self._tool_result(message.body))
        console.print()

    def _when(self, moment: Optional[datetime]) -> str:
        """Render a stored timestamp in the reader's own clock.

        SQLite hands back what was written — UTC, with the offset dropped — so
        a naive value is stamped as UTC before it is converted, and a wall
        clock two hours off is avoided.
        """
        if moment is None:
            return "—"
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        return moment.astimezone().strftime(TIMESTAMP_FORMAT)

    def notice(self, text: str) -> None:
        self._console.print(
            self._marked(NOTICE_MARKER, "core.muted", Text(text, style="core.muted"))
        )

    def error(self, text: str) -> None:
        self._console.print(
            self._marked(ERROR_MARKER, "core.danger", Text(text, style="core.danger"))
        )

    def help(self) -> None:
        console = self._console
        table = Table.grid(padding=(0, 2))
        table.add_column(style="core.command", no_wrap=True)
        table.add_column(style="core.muted")
        for command in self._commands:
            table.add_row(command.name, command.summary)
        console.print()
        console.print(table)
        console.print()

    def clear(self) -> None:
        self._console.clear()

    def farewell(self) -> None:
        self._console.print(Text(f"{NAME} session ended.", style="core.faint"))

    def _marked(self, marker: str, marker_style: str, body: RenderableType) -> Table:
        """Lay a one-character gutter beside a block, keeping wrapped lines aligned."""
        row = Table.grid(padding=(0, 1))
        row.add_column(width=1, no_wrap=True)
        row.add_column(overflow="fold")
        row.add_row(Text(marker, style=marker_style), body)
        return row
