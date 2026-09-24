"""Where the chat model comes from.

One function, `build_model`, turns the configured provider into a live model.
It dispatches explicitly rather than through a registry keyed by the enum: a
registry would force every factory to accept the same constructor arguments,
and they genuinely differ — `num_ctx` means nothing to a hosted service, an API
key means nothing to a daemon on localhost. Three branches say that honestly.
"""

from typing import Optional

from langchain_core.language_models import BaseChatModel

from src.services.agent.providers.anthropic import AnthropicFactory
from src.services.agent.providers.ollama import OllamaFactory
from src.services.agent.providers.openai import OpenaiFactory
from src.services.agent.providers.provider import ProviderFactory
from src.utils.enums.provider import Provider


def build_model(
    *,
    provider: Provider,
    model: str,
    api_key: Optional[str],
    num_ctx: int,
) -> BaseChatModel:
    """Build the chat model named by configuration."""
    if provider is Provider.OLLAMA:
        return OllamaFactory(model=model, num_ctx=num_ctx).factory_method()

    if provider is Provider.OPENAI:
        return OpenaiFactory(
            model=model,
            api_key=_credential(api_key, provider),
        ).factory_method()

    return AnthropicFactory(
        model=model,
        api_key=_credential(api_key, provider),
    ).factory_method()


def _credential(api_key: Optional[str], provider: Provider) -> str:
    """Insist on a key for the providers that charge for the call.

    `Settings` already refuses to load without one, so reaching this is a
    programming error rather than a configuration mistake — but the type is
    `Optional` and the narrowing has to happen somewhere.
    """
    if not api_key:
        raise ValueError(f"Provider {provider.value} requires an API key.")
    return api_key


__all__ = ["build_model", "ProviderFactory"]
