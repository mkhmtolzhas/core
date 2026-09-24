"""Repository for recorded session turns."""

from abc import ABC, abstractmethod
from functools import lru_cache
from typing import List
from uuid import UUID

from sqlalchemy import select

from src.core.connections.database.sqlalchemy_connection import sqlalchemy_connection
from src.entities.message import Message
from src.models.message import MessageModel
from src.repos.repository import IRepository, Repository
from src.utils.enums.message_role import MessageRole


class MessageRepository(IRepository[Message], ABC):
    """Repository contract for recorded session turns."""

    @abstractmethod
    async def history(self, session_id: UUID) -> List[Message]:
        """Return one session's turns in the order they happened.

        Unpaginated on purpose: a transcript read back is read whole, and a
        session that outgrows memory has bigger problems than this query.
        Callers that want a window use the generic ``find`` instead.
        """


class MessageRepositoryImpl(Repository[MessageModel, Message], MessageRepository):
    """SQLAlchemy repository for recorded session turns."""

    def _model_to_entity(self, model: MessageModel) -> Message:
        return Message(
            id=model.id,
            session_id=model.session_id,
            role=MessageRole(model.role),
            body=model.body,
            tool_name=model.tool_name,
            tool_arguments=model.tool_arguments,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )

    async def history(self, session_id: UUID) -> List[Message]:
        async with self.connection_pool() as session:
            stmt = (
                select(MessageModel)
                .where(MessageModel.session_id == session_id)
                .order_by(MessageModel.created_at.asc())
            )
            rows = (await session.execute(stmt)).scalars().all()
            return [self._model_to_entity(row) for row in rows]


@lru_cache(maxsize=1)
def get_message_repository() -> MessageRepository:
    """Return the shared message repository."""

    return MessageRepositoryImpl(
        connection=sqlalchemy_connection, model=MessageModel, entity=Message
    )
