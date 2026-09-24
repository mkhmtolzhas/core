from langchain.agents.middleware import ModelCallLimitMiddleware


def build(*, thread_limit: int, run_limit: int) -> ModelCallLimitMiddleware:
    """Cap how many times the agent may call the model.

    Runs as a `before_model` hook, so it is placed first in the chain: it
    short-circuits a runaway loop before any context or model work is paid for.
    `thread_limit` only applies when the agent is built with a checkpointer.
    """
    return ModelCallLimitMiddleware(
        thread_limit=thread_limit,
        run_limit=run_limit,
        exit_behavior="end",
    )
