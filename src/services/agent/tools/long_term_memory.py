"""Tools that let the agent carry facts across sessions.

The docstrings are not documentation for us: they are the only description the
model reads when it decides whether to call a tool, so they say *when* to reach
for one rather than merely what it does.
"""

import uuid

from langchain_core.tools import tool
from langgraph.prebuilt.tool_node import ToolRuntime
from langgraph.store.base import BaseStore

from src.services.agent.memory.long_term import (
    MEMORY_FIELD,
    MEMORY_NAMESPACE,
)
from src.services.agent.types import Context

SEARCH_LIMIT = 5


def _store(runtime: ToolRuntime[Context]) -> BaseStore:
    """Fail loudly when the agent was built without a store.

    `runtime.store` is `None` unless `create_agent` was given one. Left
    unchecked it surfaces as `AttributeError: 'NoneType'` from inside the tool,
    which the agent reports to the model as a tool failure and then tries to
    work around — hiding a wiring mistake as flaky behaviour.
    """
    store = runtime.store
    if store is None:
        raise RuntimeError(
            "Long-term memory is unavailable: the agent was built without a store."
        )
    return store


@tool
async def get_lt_context(runtime: ToolRuntime[Context], query: str) -> str:
    """Search long-term memory for facts remembered in earlier sessions.

    Use this before answering anything that depends on the user's preferences,
    past decisions, or project conventions. Phrase `query` as the thing you want
    to recall, in the user's own words where possible.
    """
    records = await _store(runtime).asearch(
        MEMORY_NAMESPACE,
        query=query,
        limit=SEARCH_LIMIT,
    )
    if not records:
        return f"No long-term context found for: {query}"

    remembered = "\n".join(f"- {record.value[MEMORY_FIELD]}" for record in records)
    return f"Long-term context for {query}:\n{remembered}"


@tool
async def save_lt_context(runtime: ToolRuntime[Context], memory: str) -> str:
    """Remember one durable fact for future sessions.

    Use this for things that stay true beyond the current task: the user's
    preferences, project conventions, decisions and their reasons. Do not use it
    for the contents of the conversation itself. Write `memory` as a single
    self-contained sentence, since it will be read back without its context.
    """
    await _store(runtime).aput(
        MEMORY_NAMESPACE,
        key=str(uuid.uuid4()),
        value={MEMORY_FIELD: memory},
    )
    return "Saved to long-term memory."
