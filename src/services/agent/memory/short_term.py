from abc import ABC, abstractmethod
from functools import lru_cache

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Checkpointer

from src.core.connections.database.database_connection import DatabaseConnection
from src.core.connections.database.sqlite_connection import sqlite_connection


class ShortTermMemory(ABC):
    """Factory for the thread memory an agent run reads and writes."""

    @abstractmethod
    def create_short_term_memory(self) -> Checkpointer:
        raise NotImplementedError()

    @abstractmethod
    async def setup(self) -> None:
        """Create whatever the backend needs before the first run."""
        raise NotImplementedError()


class SqliteShortTermMemory(ShortTermMemory):
    """Checkpointer backed by the shared SQLite connection.

    Takes the connection rather than a path so the live handle is owned by one
    place: the checkpointer, the store and anything else added later then share
    a single transaction context instead of racing each other over the file.

    `AsyncSqliteSaver` is built straight from that handle. Its
    `from_conn_string` sibling is an async context manager that opens *and
    closes* the database around a `with` block, which cannot express a
    connection living as long as the session.
    """

    def __init__(self, database_connection: DatabaseConnection) -> None:
        self._database_connection = database_connection

    def _saver(self) -> AsyncSqliteSaver:
        return AsyncSqliteSaver(self._database_connection.connection_factory())

    def create_short_term_memory(self) -> Checkpointer:
        return self._saver()

    async def setup(self) -> None:
        await self._saver().setup()


@lru_cache(maxsize=1)
def get_short_term_memory() -> ShortTermMemory:
    return SqliteShortTermMemory(database_connection=sqlite_connection)
