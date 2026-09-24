from abc import ABC, abstractmethod

from src.core.connections.connection import Connection


class DatabaseConnection(Connection, ABC):
    @abstractmethod
    def connection_factory(self):
        """Return the live handle every consumer of this database talks to.

        What a handle is depends on the driver: a shared `aiosqlite.Connection`
        for the raw one, an `async_sessionmaker` for the ORM. Callers name what
        they need by choosing a connection, not by inspecting what comes back.
        """
        raise NotImplementedError(
            "Subclasses must implement connection_factory method"
        )
