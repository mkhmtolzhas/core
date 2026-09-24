"""Domain entities for CLI sessions."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from src.entities.entity import Entity
from src.entities.mixins.database import DatabaseMixin
from src.utils.enums.provider import Provider


@dataclass(kw_only=True)
class Session(Entity, DatabaseMixin):
    """One conversation, kept for as long as the database keeps it.

    `thread_id` is the same id LangGraph files its checkpoints under, which is
    what makes this row a key into that state rather than a copy of it: the
    metadata here says which model answered and in which workspace, and the
    checkpointer still owns the graph state itself.

    A session has no end. It is left, and it can be returned to — from a later
    prompt, a later launch of the CLI or a different terminal — so there is no
    closing stamp to write and none to disbelieve after a crash.
    """

    thread_id: str
    provider: Provider
    model: str
    workspace_root: str


@dataclass(kw_only=True)
class SessionDigest(Entity):
    """A session as it appears in a list of sessions to choose from.

    Everything here except the session itself is derived from its messages, and
    derived on read rather than kept in a column: a title that is stored has to
    be written, corrected and migrated, whereas the first thing the user typed
    is already the best name the session will ever have.
    """

    session: Session
    preview: str
    message_count: int
    last_activity: Optional[datetime] = None

    @property
    def activity(self) -> Optional[datetime]:
        """When the session was last spoken to, or opened if it never was."""
        return self.last_activity or self.session.created_at
