from langchain_openai import ChatOpenAI

from src.services.agent.providers.provider import ProviderFactory


class OpenaiFactory(ProviderFactory):
    """A model hosted by OpenAI."""

    def __init__(self, model: str, api_key: str) -> None:
        self._model = model
        self._api_key = api_key

    def factory_method(self) -> ChatOpenAI:
        return ChatOpenAI(model=self._model, api_key=self._api_key)  # type: ignore[arg-type]
