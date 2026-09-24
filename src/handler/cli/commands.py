"""Slash commands of the terminal.

Commands are pure data plus a resolver. Deciding *what* a command means is
kept here; actually clearing the screen, switching sessions or leaving the
loop is the handler's job, so this module stays free of side effects and easy
to extend.
"""

from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple, Optional, Tuple

COMMAND_PREFIX = "/"


class CommandAction(Enum):
    """What the handler should do once a command is recognised."""

    HELP = "help"
    CLEAR = "clear"
    SESSIONS = "sessions"
    RESUME = "resume"
    NEW = "new"
    EXIT = "exit"


@dataclass(frozen=True)
class Command:
    """One entry of the slash-command palette."""

    name: str
    summary: str
    action: CommandAction


class Invocation(NamedTuple):
    """A recognised command line, split into what to do and what with.

    The argument is kept as the raw remainder of the line rather than parsed
    here: what counts as a valid one is the handler's business, and only it can
    say whether `3` names a session.
    """

    action: CommandAction
    argument: str


COMMANDS: Tuple[Command, ...] = (
    Command("/help", "show the available commands", CommandAction.HELP),
    Command("/clear", "clear the screen", CommandAction.CLEAR),
    Command("/sessions", "list saved sessions", CommandAction.SESSIONS),
    Command("/resume", "reopen a session: /resume 2", CommandAction.RESUME),
    Command("/new", "start a fresh session", CommandAction.NEW),
    Command("/exit", "end the session", CommandAction.EXIT),
    Command("/quit", "end the session", CommandAction.EXIT),
)


def is_command(text: str) -> bool:
    """True when the line should be routed to a command instead of the agent."""
    return text.lstrip().startswith(COMMAND_PREFIX)


def resolve(text: str) -> Optional[Invocation]:
    """Map a typed line to its action and argument, or `None` when unknown."""
    name, _, argument = text.strip().partition(" ")
    name = name.lower()
    for command in COMMANDS:
        if command.name == name:
            return Invocation(command.action, argument.strip())
    return None
