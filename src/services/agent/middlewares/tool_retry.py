from langchain.agents.middleware import ToolRetryMiddleware


def build(
    *,
    max_retries: int,
    backoff_factor: float,
    initial_delay: float,
) -> ToolRetryMiddleware:
    """Retry flaky tool calls before surfacing anything to the model.

    Sits inner-most in the tool chain with `on_failure="error"` so that a
    definitively failed call is raised to `ToolErrorMiddleware`, which decides
    what the model gets to see.
    """
    return ToolRetryMiddleware(
        max_retries=max_retries,
        backoff_factor=backoff_factor,
        initial_delay=initial_delay,
        on_failure="error",
    )
