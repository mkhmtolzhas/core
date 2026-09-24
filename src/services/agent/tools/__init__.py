from typing import Tuple

from langchain.tools import BaseTool

from src.services.agent.tools.long_term_memory import (
    get_lt_context,
    save_lt_context,
)


def get_tools() -> Tuple[BaseTool, ...]:
    """Custom tools for the agent.

    Tools that come from middleware (filesystem, shell, todos) are registered by
    that middleware and must not be repeated here.
    """
    return (get_lt_context, save_lt_context)
