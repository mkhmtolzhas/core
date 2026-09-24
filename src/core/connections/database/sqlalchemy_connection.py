"""SQLAlchemy async connection — the source of truth for ORM CRUD.

LangGraph's checkpointer and long-term store live on a separate connection
(`sqlite_connection`), because they take a raw `aiosqlite.Connection` and know
nothing about sessions or mapped classes. This one only serves the session
factory the generic repository consumes. Two handles, one file: WAL keeps a
reader running while the other side writes, and `busy_timeout` makes a
contended write wait rather than raise `database is locked`.
"""

from typing import Any, Optional

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.core.config import settings
from src.core.connections.database.database_connection import DatabaseConnection
from src.core.loggers.file_logger import file_logger


class SqlAlchemyConnection(DatabaseConnection):
    """Async engine plus the session factory built on it.

    A pool is right here in a way it is not for the raw handle: every session
    checks out a connection of its own, so its transaction is its own, and the
    engine hands it back when the `async with` block ends. What the raw
    connection cannot afford — several handles sharing one transaction state —
    is exactly what SQLAlchemy needs to keep units of work separate.
    """

    def __init__(self, db_url: str, busy_timeout_ms: int = 5000) -> None:
        self._db_url = db_url
        self._busy_timeout_ms = busy_timeout_ms
        self.engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker[AsyncSession]] = None

    def _apply_pragmas(self, dbapi_connection: Any, _record: Any) -> None:
        """Set the per-connection PRAGMAs every pooled handle needs.

        They are connection state, not database state: `journal_mode` persists
        in the file, but `busy_timeout` and `foreign_keys` reset to their
        defaults on every new handle. The pool opens handles as it grows, so
        the only place that sees all of them is this event.
        """
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute(f"PRAGMA busy_timeout={int(self._busy_timeout_ms)}")
            cursor.execute("PRAGMA foreign_keys=ON")
        finally:
            cursor.close()

    def _ensure_initialized(self) -> None:
        if self.engine is not None and self._session_factory is not None:
            return

        self.engine = create_async_engine(self._db_url, echo=False)
        event.listens_for(self.engine.sync_engine, "connect")(self._apply_pragmas)
        self._session_factory = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    def connection_factory(self) -> async_sessionmaker[AsyncSession]:
        """Return the session factory repositories open their units of work on."""
        self._ensure_initialized()
        if self._session_factory is None:
            raise RuntimeError("SQLAlchemy session factory is not initialized.")
        return self._session_factory

    async def connect(self) -> None:
        """Build the engine and prove the database answers.

        The engine itself is lazy — it opens nothing until the first query — so
        without this round trip a bad path or an unreadable file would surface
        much later, as a failed write in the middle of a session.
        """
        self._ensure_initialized()
        if self.engine is None:
            raise RuntimeError("SQLAlchemy engine is not initialized.")
        try:
            async with self.engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
        except Exception as error:
            file_logger.error(f"Failed to connect to the database: {error}")
            raise

    async def close(self) -> None:
        if self.engine is None:
            return
        try:
            await self.engine.dispose()
        except Exception as error:
            file_logger.error(f"Failed to disconnect from the database: {error}")
            raise
        finally:
            self.engine = None
            self._session_factory = None


sqlalchemy_connection = SqlAlchemyConnection(
    db_url=settings.db_async_url(),
    busy_timeout_ms=settings.db_busy_timeout_ms,
)
