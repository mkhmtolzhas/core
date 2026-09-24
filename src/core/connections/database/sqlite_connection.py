from typing import Optional

import aiosqlite

from src.core.config import settings
from src.core.connections.database.database_connection import DatabaseConnection


class SqliteConnection(DatabaseConnection):
    """SQLite database connection class.

    Deliberately not a pool. SQLite is a library inside this process, not a
    server: a connection is a file handle plus a page cache, so pooling buys
    no concurrency (writes serialise on the file either way) and costs
    correctness, since each handle would carry its own transaction state over
    the same database. One connection is opened for the process and shared by
    every consumer.
    """

    def __init__(
        self,
        db_path: str,
        busy_timeout_ms: int = 5000,
    ) -> None:
        self._db_path = db_path
        self._busy_timeout_ms = busy_timeout_ms
        self._connection: Optional[aiosqlite.Connection] = None

    async def connect(self) -> None:
        """Open the connection in autocommit mode.

        `isolation_level=None` turns off the implicit transaction the DB-API
        layer would otherwise open before the first write and hold until an
        explicit commit. With the default, a component that issues its own
        `BEGIN` on this shared connection fails with `cannot start a
        transaction within a transaction` — LangGraph's SQLite store does
        exactly that. Autocommit hands transaction control to whoever owns the
        statement, which is what every consumer here already assumes.

        WAL lets a reader run while a write is in flight, and `busy_timeout`
        makes a contended write wait instead of raising `database is locked`
        the moment a second process touches the file.
        """

        if self._connection is not None:
            return

        connection = await aiosqlite.connect(self._db_path, isolation_level=None)
        await connection.execute("PRAGMA journal_mode=WAL")
        await connection.execute(f"PRAGMA busy_timeout={int(self._busy_timeout_ms)}")
        self._connection = connection

    async def close(self) -> None:
        """
        Close the SQLite database connection.

        Mandatory, not merely tidy: `aiosqlite` drives the C library from a
        dedicated worker thread that is not a daemon, so an unclosed connection
        keeps the interpreter alive and the CLI never returns to the shell.
        """

        if self._connection is None:
            return

        await self._connection.close()
        self._connection = None

    def connection_factory(self) -> aiosqlite.Connection:
        if self._connection is None:
            raise RuntimeError("SQLite connection is not initialized.")
        return self._connection


sqlite_connection = SqliteConnection(
    db_path=str(settings.db_path()),
    busy_timeout_ms=settings.db_busy_timeout_ms,
)
