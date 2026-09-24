"""Bring the database up to date so the agent can run.

Idempotent: the ORM tables are created with `IF NOT EXISTS`, and every LangGraph
backend records the migrations it has applied, so running this twice is a no-op.
Safe to call on every deploy, and cheap enough to call before a first local run.
"""

import asyncio

from src.core.app.lifespan import lifespan
from src.core.config import settings
from src.core.connections.database.sqlalchemy_connection import sqlalchemy_connection
from src.models import BaseModel
from src.services.agent.memory.long_term import get_long_term_memory
from src.services.agent.memory.short_term import get_short_term_memory


async def _setup_schema() -> None:
    """Create the application's own tables from the ORM metadata.

    `create_all` rather than a migration tool: the schema has one owner, one
    deployment and one file, so the cost of Alembic buys nothing yet. It stops
    being enough the first time a column has to change on a database that
    already holds sessions worth keeping.
    """
    engine = sqlalchemy_connection.engine
    if engine is None:
        raise RuntimeError("SQLAlchemy engine is not initialized.")
    async with engine.begin() as connection:
        await connection.run_sync(BaseModel.metadata.create_all)


async def setup() -> None:
    """Prepare the schema and both memories against the live connections."""
    async with lifespan():
        print(f"Setting up {settings.db_path()} ...")

        await _setup_schema()
        print("Sessions and messages tables are up to date.")

        await get_short_term_memory().setup()
        print("Short-term memory (checkpointer) is up to date.")

        await get_long_term_memory().setup()
        print(
            "Long-term memory (store) is up to date: "
            f"{settings.embedding_model}, {settings.embedding_dims} dims."
        )


def main() -> None:
    asyncio.run(setup())


if __name__ == "__main__":
    main()
