"""What the agent emits while it works, as plain data.

One turn is not one answer: the agent writes, calls a tool, reads the result
and writes again. These are the pieces of that, in the order they happen, so a
caller can show the work instead of only its conclusion.

Three separate types rather than one record with a kind flag: a consumer
dispatches on the type and the compiler tells it when a new kind of event
appears.
"""

from dataclasses import dataclass
from typing import Any, Mapping, Union

from src.entities.entity import Entity


@dataclass(kw_only=True)
class TextChunk(Entity):
    """A fragment of the answer, as small as a single token."""

    body: str


@dataclass(kw_only=True)
class ToolCall(Entity):
    """The agent decided to use a tool, before it runs."""

    name: str
    arguments: Mapping[str, Any]


@dataclass(kw_only=True)
class ToolResult(Entity):
    """What the tool returned."""

    name: str
    body: str


AgentEvent = Union[TextChunk, ToolCall, ToolResult]
