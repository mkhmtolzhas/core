from langchain_anthropic import ChatAnthropic

from src.services.agent.providers.provider import ProviderFactory


class AnthropicFactory(ProviderFactory):
    """A model hosted by Anthropic."""

    def __init__(self, model: str, api_key: str) -> None:
        self._model = model
        self._api_key = api_key

    def factory_method(self) -> ChatAnthropic:
        return ChatAnthropic(model=self._model, api_key=self._api_key)  # type: ignore[call-arg]
