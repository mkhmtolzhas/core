from contextlib import asynccontextmanager

from src.core.connections.connection import Connection
from src.core.connections.database.sqlalchemy_connection import sqlalchemy_connection
from src.core.connections.database.sqlite_connection import sqlite_connection
from src.core.connections.observability.langfuse_connection import (
    langfuse_connection,
)
from src.core.loggers.file_logger import file_logger


@asynccontextmanager
async def lifespan():
    """Own every connection for the length of one CLI session.

    Nothing is opened at import time: a connection is a live resource, and the
    module that declares it must not decide when it starts. Shutdown runs in a
    `finally` so a crashing session still closes the database — otherwise the
    process would hang on the way out instead of reporting the error.

    Two connections to the same file, closed in reverse order of opening. The
    raw handle serves LangGraph's checkpointer and store; the SQLAlchemy engine
    serves the repositories. Neither can stand in for the other.

    Langfuse opens first and closes last, so it outlives everything it might
    trace — and closing it is what sends the final prompt's trace.
    """
    file_logger.info("Starting up the application...")
    await startup(langfuse_connection)
    await startup(sqlite_connection)
    await startup(sqlalchemy_connection)
    file_logger.info("Application started up successfully.")
    try:
        yield
    finally:
        file_logger.info("Shutting down the application...")
        await shutdown(sqlalchemy_connection)
        await shutdown(sqlite_connection)
        await shutdown(langfuse_connection)
        file_logger.info("Application shut down successfully.")


async def startup(connection: Connection):
    await connection.connect()


async def shutdown(connection: Connection):
    await connection.close()
