"""Prompts the agent runs on, and where their text comes from.

Managed in Langfuse when it is configured, so a prompt can be edited and
versioned without a deploy; fixed in code otherwise. The text in code is also
the fallback when Langfuse cannot serve a prompt, so the agent answers the
same way whether or not a prompt store is reachable.
"""

from functools import lru_cache

from src.core.config import settings
from src.core.connections.observability.langfuse_connection import (
    langfuse_connection,
)
from src.core.loggers.file_logger import file_logger
from src.services.agent.prompts.langfuse_source import LangfusePromptSource
from src.services.agent.prompts.source import PromptSource, StaticPromptSource
from src.services.agent.prompts.system import SYSTEM_PROMPT, SYSTEM_PROMPT_NAME


@lru_cache(maxsize=1)
def get_system_prompt_source() -> PromptSource:
    if not langfuse_connection.enabled:
        return StaticPromptSource(SYSTEM_PROMPT)

    return LangfusePromptSource(
        connection=langfuse_connection,
        name=SYSTEM_PROMPT_NAME,
        label=settings.langfuse_prompt_label,
        fallback=SYSTEM_PROMPT,
        logger=file_logger,
    )


async def warm_up() -> None:
    """Fetch every managed prompt once, so the first turn finds it cached.

    Meant to run in the background while the user types: the one fetch that
    has to wait for the network then happens before anyone is waiting on it.
    """
    await get_system_prompt_source().get()


__all__ = ["get_system_prompt_source", "warm_up", "PromptSource"]
