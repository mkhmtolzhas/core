from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Sequence, Tuple, cast

from langchain.embeddings.base import Embeddings
from langchain_ollama import OllamaEmbeddings
from langgraph.store.base import BaseStore
from langgraph.store.sqlite import AsyncSqliteStore
from langgraph.store.sqlite.base import SqliteIndexConfig

from src.core.config import settings
from src.core.connections.database.database_connection import DatabaseConnection
from src.core.connections.database.sqlite_connection import sqlite_connection

MEMORY_NAMESPACE: Tuple[str, ...] = ("memory",)
"""Bucket every remembered fact is filed under.

The namespace answers *where* a record is kept, not what is inside it. It is
given on each call rather than configured once, so the same store can hold
several unrelated collections side by side.
"""

MEMORY_FIELD = "memory"
"""Key under which a remembered fact is stored inside a record.

Written by the save tool, read by the retrieve tool, and indexed below. All
three have to agree, which is why the name is defined once, here.
"""

MEMORY_INDEX_FIELDS: Tuple[str, ...] = (MEMORY_FIELD,)
"""Keys of a stored record whose text is embedded and made searchable.

Not to be confused with the namespace, which says *where* a record lives and is
given per call. This list says which keys *inside* one record are turned into a
vector: everything else is still stored and returned in full, it just cannot be
matched against a query.
"""


class LongTermMemory(ABC):
    """Factory for the memory that outlives a single thread."""

    @abstractmethod
    def create_long_term_memory(self) -> BaseStore:
        raise NotImplementedError()

    @abstractmethod
    async def setup(self) -> None:
        """Create whatever the backend needs before the first run."""
        raise NotImplementedError()


class SqliteLongTermMemory(LongTermMemory):
    """Store backed by the shared SQLite connection, with vector search.

    No separate vector database: the store keeps its embeddings in the same
    file through `sqlite-vec`, so a semantic search and the record it returns
    never disagree about what is stored.

    The embedding model is injected rather than built here, so a test can hand
    in a deterministic stub instead of reaching for Ollama.
    """

    def __init__(
        self,
        database_connection: DatabaseConnection,
        embedding_model: Embeddings,
        embedding_dims: int,
        index_fields: Sequence[str],
    ) -> None:
        self._database_connection = database_connection
        self._embedding_model = embedding_model
        self._embedding_dims = embedding_dims
        self._index_fields = index_fields

    def _store(self) -> AsyncSqliteStore:
        """Build the store.

        Callable only from inside a running event loop: `AsyncSqliteStore`
        binds itself to the loop it is constructed on, so building it at import
        time would tie the store to a loop that no longer exists by the time a
        request arrives.
        """
        fields = list(self._index_fields)
        index = cast(
            SqliteIndexConfig,
            {
                "dims": self._embedding_dims,
                "embed": self._embedding_model,
                "fields": fields,
                "text_fields": fields,
            },
        )
        return AsyncSqliteStore(
            self._database_connection.connection_factory(),
            index=index,
        )

    def create_long_term_memory(self) -> BaseStore:
        return self._store()

    async def setup(self) -> None:
        await self._store().setup()


@lru_cache(maxsize=1)
def get_long_term_memory() -> LongTermMemory:
    return SqliteLongTermMemory(
        database_connection=sqlite_connection,
        embedding_model=OllamaEmbeddings(model=settings.embedding_model),
        embedding_dims=settings.embedding_dims,
        index_fields=MEMORY_INDEX_FIELDS,
    )
