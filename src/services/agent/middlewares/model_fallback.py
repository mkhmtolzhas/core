from typing import Optional, Sequence

from langchain.agents.middleware import ModelFallbackMiddleware


def build(*, fallback_models: Sequence[str]) -> Optional[ModelFallbackMiddleware]:
    """Switch models when the primary one is exhausted or down.

    Wraps the model call from outside `ModelRetryMiddleware`, so a fallback is
    only attempted after retries on the primary model are used up. Returns
    `None` when no fallback models are configured.
    """
    if not fallback_models:
        return None

    first, *rest = fallback_models
    return ModelFallbackMiddleware(first, *rest)
