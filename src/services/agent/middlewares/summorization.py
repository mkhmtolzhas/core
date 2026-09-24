from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.language_models import BaseChatModel


def build(
    *,
    model: BaseChatModel,
    trigger_tokens: int,
    keep_messages: int,
) -> SummarizationMiddleware:
    """Compact the history with an LLM when trimming is no longer enough.

    Takes the agent's own chat model instance so summarization runs on the same
    local Ollama endpoint instead of resolving a second provider from a string.
    """
    return SummarizationMiddleware(
        model=model,
        trigger=("tokens", trigger_tokens),
        keep=("messages", keep_messages),
    )
