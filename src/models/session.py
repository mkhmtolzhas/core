"""SQLAlchemy model for one CLI session."""

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from src.models.base_model import BaseModel


class SessionModel(BaseModel):
    """Persistence model for a session's metadata.

    `created_at` from the base model is the moment the session opened, so no
    separate `started_at` column exists to disagree with it. There is no
    closing timestamp either: a session is resumable, so the last message is
    what says when it was last used, and that is a fact about the messages.
    """

    __tablename__ = "sessions"

    thread_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    provider: Mapped[str] = mapped_column(String(32), index=True)
    model: Mapped[str] = mapped_column(String(128), index=True)
    workspace_root: Mapped[str] = mapped_column(String(1024))
