"""SQLAlchemy model for one recorded turn of a session."""

from typing import Any, Optional
from uuid import UUID

from sqlalchemy import JSON, ForeignKey, String, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from src.models.base_model import BaseModel


class MessageModel(BaseModel):
    """Persistence model for a message, tool call or tool result.

    The role is a plain string rather than a database enum: a new role should
    cost a line in `MessageRole` and nothing else, whereas a native enum type
    would need a migration to add one. The entity still carries the enum — the
    repository converts on the way out.
    """

    __tablename__ = "messages"

    session_id: Mapped[UUID] = mapped_column(
        Uuid(),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        index=True,
    )
    role: Mapped[str] = mapped_column(String(16), index=True)
    body: Mapped[str] = mapped_column(Text, default="")
    tool_name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    tool_arguments: Mapped[Optional[dict[str, Any]]] = mapped_column(
        JSON, nullable=True
    )
