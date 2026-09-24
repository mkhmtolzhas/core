from langchain_ollama import ChatOllama

from src.services.agent.providers.provider import ProviderFactory


class OllamaFactory(ProviderFactory):
    """A model served by a local Ollama daemon. Needs no credentials."""

    def __init__(self, model: str, num_ctx: int) -> None:
        self._model = model
        self._num_ctx = num_ctx

    def factory_method(self) -> ChatOllama:
        """`num_ctx` is not optional in practice.

        Ollama defaults to 4096 tokens, and the system prompt plus the tool
        schemas already claim most of that — leaving too little for the
        conversation itself and making the agent answer in fragments.
        """
        return ChatOllama(model=self._model, num_ctx=self._num_ctx)
