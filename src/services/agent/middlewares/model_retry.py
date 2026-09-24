from langchain.agents.middleware import ModelRetryMiddleware


def build(
    *,
    max_retries: int,
    backoff_factor: float,
    initial_delay: float,
    raise_on_failure: bool,
) -> ModelRetryMiddleware:
    """Retry transient model failures with exponential backoff.

    Set `raise_on_failure` only when a fallback model is configured: the
    exhausted call then propagates outward to `ModelFallbackMiddleware`. With no
    fallback there is nothing to escalate to, and injecting the error as a
    message lets the agent report it instead of crashing the run.
    """
    return ModelRetryMiddleware(
        max_retries=max_retries,
        backoff_factor=backoff_factor,
        initial_delay=initial_delay,
        on_failure="error" if raise_on_failure else "continue",
    )
