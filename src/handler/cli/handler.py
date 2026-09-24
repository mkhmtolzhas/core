"""Terminal handler: the read-ask-render loop of CORE.

The handler is a plug into the use-case layer. It knows the `AgentUseCase`
abstraction and the DI function that produces one, never the implementation
behind it, and it speaks to that layer only in entities.
"""

from abc import ABC, abstractmethod
from functools import lru_cache
from typing import AsyncIterator, Callable, Optional, Sequence

from src.entities.event import AgentEvent
from src.entities.prompt import Prompt
from src.entities.session import SessionDigest
from src.handler.cli.commands import COMMANDS, CommandAction, is_command, resolve
from src.handler.cli.console import build_console
from src.handler.cli.paths import get_history_path, get_workspace
from src.handler.cli.prompt import PromptReader, TerminalPromptReader
from src.handler.cli.render import Renderer, TerminalRenderer
from src.handler.cli.status import Interrupted, interruptible
from src.handler.cli.theme import (
    ACCENT,
    PROMPT_STYLE,
    REFRESH_RATE,
    SPINNER_FRAMES,
)
from src.usecase.agent.usecase import AgentUseCase, get_agent_use_case

PROMPT_MARKER = "› "
INPUT_HINT = "enter to send · alt+enter for a new line · /help for commands"
THINKING_LABEL = "Thinking"
THINKING_HINT = "ctrl+c to interrupt"
TOOL_ARGUMENT_LIMIT = 60
"""Characters of a tool argument kept on the call line."""

TOOL_RESULT_LIMIT = 100
"""Characters of a tool result kept on its summary line."""

SESSION_LIST_LIMIT = 20
"""Sessions offered by `/sessions`, and the range `/resume <n>` numbers.

A window rather than the whole history: past the last twenty, a person looks
for a session by what was said in it, not by scrolling — and that is a search,
which is a different command.
"""


class CliHandler(ABC):
    """Entry point of the terminal handler."""

    @abstractmethod
    async def run(self) -> None:
        """Drive the session until the user leaves."""
        raise NotImplementedError()


class CliHandlerImpl(CliHandler):
    def __init__(
        self,
        *,
        agent_use_case_factory: Callable[[], AgentUseCase],
        reader: PromptReader,
        renderer: Renderer,
        workspace: str,
    ) -> None:
        self._agent_use_case_factory = agent_use_case_factory
        self._agent_use_case: Optional[AgentUseCase] = None
        self._reader = reader
        self._renderer = renderer
        self._workspace = workspace

    @property
    def agent_use_case(self) -> AgentUseCase:
        """Built on first use, so the banner appears before the model loads."""
        if self._agent_use_case is None:
            self._agent_use_case = self._agent_use_case_factory()
        return self._agent_use_case

    async def run(self) -> None:
        self._renderer.banner(self._workspace)

        while True:
            try:
                line = await self._reader.read()
            except (EOFError, KeyboardInterrupt):
                break

            if line is None:
                break

            text = line.strip()
            if not text:
                continue

            self._renderer.prompt_echo(text)

            if is_command(text):
                if not await self._run_command(text):
                    break
                continue

            await self._ask(text)

        self._renderer.farewell()

    async def _run_command(self, text: str) -> bool:
        """Execute a slash command; `False` means the CLI should be left.

        Leaving is not ending: every conversation stays in the database with
        the thread it was checkpointed on, and `/resume` picks any of them up
        again — in this run or in a later one.
        """
        invocation = resolve(text)

        if invocation is None:
            name = text.split(maxsplit=1)[0]
            self._renderer.error(f"Unknown command {name}. Try /help.")
            return True

        action, argument = invocation

        if action is CommandAction.HELP:
            self._renderer.help()
        elif action is CommandAction.CLEAR:
            self._renderer.clear()
            self._renderer.banner(self._workspace)
        elif action is CommandAction.SESSIONS:
            await self._list_sessions()
        elif action is CommandAction.RESUME:
            await self._resume_session(argument)
        elif action is CommandAction.NEW:
            await self._start_new_session()
        elif action is CommandAction.EXIT:
            return False

        return True

    async def _list_sessions(self) -> None:
        digests = await self.agent_use_case.recent_sessions(limit=SESSION_LIST_LIMIT)
        self._renderer.sessions(digests, self.agent_use_case.current_session_id)

    async def _start_new_session(self) -> None:
        self.agent_use_case.start_new_session()
        self._renderer.clear()
        self._renderer.banner(self._workspace)
        self._renderer.notice("New session. The previous one is saved.")

    async def _resume_session(self, argument: str) -> None:
        """Reopen a session and put it back on the screen.

        The screen is cleared first: the point of resuming is to be inside that
        conversation, and leaving the previous one above it would make the
        replayed history read as a continuation of something else.
        """
        digests = await self.agent_use_case.recent_sessions(limit=SESSION_LIST_LIMIT)

        if not digests:
            self._renderer.notice("No saved sessions yet. Ask something to start one.")
            return

        if not argument:
            self._renderer.sessions(digests, self.agent_use_case.current_session_id)
            return

        digest = self._select_session(argument, digests)
        if digest is None:
            self._renderer.error(
                f"No session matches {argument!r}. Try /sessions for the list."
            )
            return

        session = digest.session
        if session.id is None:
            self._renderer.error("That session has no id to reopen it by.")
            return

        messages = await self.agent_use_case.resume_session(session.id)
        self._renderer.clear()
        self._renderer.banner(
            self._workspace, session=digest.preview or session.thread_id
        )
        self._renderer.transcript(messages)

    def _select_session(
        self, argument: str, digests: Sequence[SessionDigest]
    ) -> Optional[SessionDigest]:
        """Resolve what the user typed against the list they were shown.

        A position first, because that is what the list offers; failing that,
        the start of a session or thread id, so a value copied out of the
        database or the log works without counting rows.
        """
        if argument.isdigit():
            position = int(argument)
            if 1 <= position <= len(digests):
                return digests[position - 1]
            return None

        needle = argument.lower()
        for digest in digests:
            identifier = str(digest.session.id or "").lower()
            if identifier.startswith(needle) or digest.session.thread_id.startswith(
                needle
            ):
                return digest
        return None

    async def _ask(self, body: str) -> None:
        try:
            await self._renderer.stream_turn(interruptible(self._answer_events(body)))
        except Interrupted:
            self._renderer.notice("Interrupted.")
        except Exception as error:
            # One bad run must not end the session. Whatever the renderer has
            # already put on screen stays there: the work really did happen,
            # and hiding it would make the failure harder to place.
            self._renderer.error(f"{type(error).__name__}: {error}")

    async def _answer_events(self, body: str) -> AsyncIterator[AgentEvent]:
        """Resolve the use case inside the generator.

        A generator body does not run until it is first advanced, which happens
        after the renderer is already drawing — so the slow first build of the
        agent sits behind the status line instead of in front of it.
        """
        async for event in self.agent_use_case.stream_answer(Prompt(body=body)):
            yield event


@lru_cache(maxsize=1)
def get_cli_handler() -> CliHandler:
    """Composition root of the handler layer.

    Collaborators are assembled here and handed down as plain values, so no
    component reaches for configuration on its own.
    """
    console = build_console()

    return CliHandlerImpl(
        agent_use_case_factory=get_agent_use_case,
        reader=TerminalPromptReader(
            marker=PROMPT_MARKER,
            history_path=get_history_path(),
            style=PROMPT_STYLE,
            commands=tuple((command.name, command.summary) for command in COMMANDS),
            hint=INPUT_HINT,
        ),
        renderer=TerminalRenderer(
            console=console,
            commands=COMMANDS,
            thinking_label=THINKING_LABEL,
            thinking_hint=THINKING_HINT,
            spinner_frames=SPINNER_FRAMES,
            spinner_style=ACCENT,
            refresh_rate=REFRESH_RATE,
            argument_limit=TOOL_ARGUMENT_LIMIT,
            result_limit=TOOL_RESULT_LIMIT,
        ),
        workspace=get_workspace(),
    )
