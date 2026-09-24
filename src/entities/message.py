"""Domain entity for one recorded turn of a session."""

from dataclasses import dataclass
from typing import Any, Mapping, Optional
from uuid import UUID

from src.entities.entity import Entity
from src.entities.mixins.database import DatabaseMixin
from src.utils.enums.message_role import MessageRole


@dataclass(kw_only=True)
class Message(Entity, DatabaseMixin):
    """What was said, by whom, and when.

    Ordering is `created_at` and nothing else. The messages of one session are
    written one after another by a single process, and the timestamp keeps
    microseconds, so the order they were stored in is the order they happened
    in — an explicit position column would only restate that, and would have to
    be kept correct by hand.

    `tool_name` and `tool_arguments` are filled only for the tool roles. The
    arguments are kept as they were passed rather than rendered into `body`,
    so a later reader can query them instead of parsing prose.
    """

    session_id: UUID
    role: MessageRole
    body: str = ""
    tool_name: Optional[str] = None
    tool_arguments: Optional[Mapping[str, Any]] = None
