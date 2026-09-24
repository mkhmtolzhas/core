from langchain.agents.middleware import TodoListMiddleware


def build() -> TodoListMiddleware:
    """Give the agent a `write_todos` tool for multi-step planning."""
    return TodoListMiddleware()
