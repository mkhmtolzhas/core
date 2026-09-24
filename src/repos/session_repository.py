"""Repository for CLI sessions."""

from abc import ABC, abstractmethod
from functools import lru_cache
from typing import List, Optional

from sqlalchemy import func, select

from src.core.connections.database.sqlalchemy_connection import sqlalchemy_connection
from src.entities.session import Session, SessionDigest
from src.models.message import MessageModel
from src.models.session import SessionModel
from src.repos.repository import IRepository, Repository
from src.utils.enums.message_role import MessageRole
from src.utils.enums.provider import Provider


class SessionRepository(IRepository[Session], ABC):
    """Repository contract for CLI sessions.

    Listing and filtering go through the generic ``find``/``count_query`` on
    the base repository, so only the two reads that need the messages table
    are declared here.
    """

    @abstractmethod
    async def get_by_thread_id(self, thread_id: str) -> Optional[Session]:
        """Return the session filed under a LangGraph thread id, or None.

        The bridge between the two halves of the database: given a thread the
        checkpointer knows about, this is what says who ran it and when.
        """

    @abstractmethod
    async def recent(self, limit: int = 20) -> List[SessionDigest]:
        """Return the most recently used sessions, freshest first.

        Ordered by the last message rather than by `created_at`, because the
        session worth offering first is the one last spoken to — an old thread
        picked up this morning belongs at the top, not where it was opened.
        """


class SessionRepositoryImpl(Repository[SessionModel, Session], SessionRepository):
    """SQLAlchemy repository for CLI sessions."""

    def _model_to_entity(self, model: SessionModel) -> Session:
        return Session(
            id=model.id,
            thread_id=model.thread_id,
            provider=Provider(model.provider),
            model=model.model,
            workspace_root=model.workspace_root,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )

    async def get_by_thread_id(self, thread_id: str) -> Optional[Session]:
        async with self.connection_pool() as session:
            stmt = select(SessionModel).where(SessionModel.thread_id == thread_id)
            row = (await session.execute(stmt)).scalar_one_or_none()
            if row is None:
                return None
            return self._model_to_entity(row)

    async def recent(self, limit: int = 20) -> List[SessionDigest]:
        """One statement, three correlated subqueries, no N+1.

        The alternative — a `GROUP BY` join — cannot also carry the first
        prompt, since an aggregate has no notion of "the earliest row's body".
        """
        preview = (
            select(MessageModel.body)
            .where(
                MessageModel.session_id == SessionModel.id,
                MessageModel.role == MessageRole.USER.value,
            )
            .order_by(MessageModel.created_at.asc())
            .limit(1)
            .correlate(SessionModel)
            .scalar_subquery()
        )
        counted = (
            select(func.count(MessageModel.id))
            .where(MessageModel.session_id == SessionModel.id)
            .correlate(SessionModel)
            .scalar_subquery()
        )
        last_activity = (
            select(func.max(MessageModel.created_at))
            .where(MessageModel.session_id == SessionModel.id)
            .correlate(SessionModel)
            .scalar_subquery()
        )

        stmt = (
            select(SessionModel, preview, counted, last_activity)
            .order_by(func.coalesce(last_activity, SessionModel.created_at).desc())
            .limit(limit)
        )

        async with self.connection_pool() as session:
            rows = (await session.execute(stmt)).all()

        return [
            SessionDigest(
                session=self._model_to_entity(model),
                preview=body or "",
                message_count=int(count or 0),
                last_activity=activity,
            )
            for model, body, count, activity in rows
        ]


@lru_cache(maxsize=1)
def get_session_repository() -> SessionRepository:
    """Return the shared session repository."""

    return SessionRepositoryImpl(
        connection=sqlalchemy_connection, model=SessionModel, entity=Session
    )
