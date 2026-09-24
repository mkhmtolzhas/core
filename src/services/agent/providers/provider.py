from abc import ABC, abstractmethod

from langchain_core.language_models.chat_models import BaseChatModel


class ProviderFactory(ABC):
    """One way of producing the chat model the agent talks through.

    Each provider takes different settings — a context window is an Ollama
    concept, an API key a hosted-service one — so the constructors deliberately
    differ. What they share is this method, which is all the caller needs.
    """

    @abstractmethod
    def factory_method(self) -> BaseChatModel:
        """Build the chat model."""
        raise NotImplementedError()
