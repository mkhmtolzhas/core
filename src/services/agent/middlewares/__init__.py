"""Middleware chain for the agent.

Order matters, and each hook type composes differently:

* `before_model` hooks run in list order;
* `after_model` hooks run in reverse list order;
* `wrap_*` hooks nest, so the first entry is the outermost wrapper.

The recommended layout below follows from that:

1. **Budget guard** — `ModelCallLimitMiddleware` first, so a runaway loop is cut
   off before any other work is done.
2. **System prompt** — `ContextMiddleware` sets the base system message from the
   run's context. Outside every tool provider: they append to the message they
   receive, so one placed outside it would have its text replaced.
3. **Tool providers** — todos, filesystem, search, shell, subagents. They only
   contribute tools and prompt text, so their relative order is cosmetic.
4. **Context management** — cheap deterministic pruning
   (`ContextEditingMiddleware`) before expensive LLM compaction
   (`SummarizationMiddleware`).
5. **Model fault tolerance** — fallback outside retry: retry the primary model,
   and only switch models once those attempts are spent.
6. **Tool fault tolerance** — error handling outside retry: retry the call, and
   only report a failure to the model once retries are spent.
"""

from typing import List, Sequence

from langchain.agents.middleware import AgentMiddleware
from langchain.tools import BaseTool
from langchain_core.language_models import BaseChatModel

from src.services.agent.middlewares import (
    context_editing,
    dynamic_prompt,
    file_search,
    filesystem,
    model_call_limit,
    model_fallback,
    model_retry,
    shell_tool,
    subagent,
    summorization,
    todo_list,
    tool_error,
    tool_retry,
)


def build_middlewares(
    *,
    model: BaseChatModel,
    workspace_root: str,
    thread_call_limit: int,
    run_call_limit: int,
    context_edit_trigger_tokens: int,
    context_edit_keep_tool_results: int,
    summarization_trigger_tokens: int,
    summarization_keep_messages: int,
    fallback_models: Sequence[str],
    model_max_retries: int,
    tool_max_retries: int,
    retry_backoff_factor: float,
    retry_initial_delay: float,
    enable_todos: bool,
    enable_filesystem: bool,
    filesystem_writable: bool,
    enable_file_search: bool,
    enable_shell: bool,
    enable_subagents: bool,
    tools: Sequence[BaseTool],
) -> List[AgentMiddleware]:
    """Assemble the enabled middlewares in execution order."""
    middlewares: List[AgentMiddleware] = [
        # 1. budget guard
        model_call_limit.build(
            thread_limit=thread_call_limit,
            run_limit=run_call_limit,
        ),  # type: ignore
        # 2. system prompt
        dynamic_prompt.build(),
    ]

    # 3. tool providers
    if enable_todos:
        middlewares.append(todo_list.build())  # type: ignore

    if enable_filesystem:
        middlewares.append(
            filesystem.build(
                workspace_root=workspace_root,
                writable=filesystem_writable,
            )  # type: ignore
        )
    if enable_file_search:
        middlewares.append(file_search.build(workspace_root=workspace_root))
    if enable_shell:
        middlewares.append(shell_tool.build(workspace_root=workspace_root))  # type: ignore
    if enable_subagents:
        middlewares.append(
            subagent.build(
                model=model,
                workspace_root=workspace_root,
                tools=tools,
                thread_call_limit=thread_call_limit,
                run_call_limit=run_call_limit,
            )  # type: ignore
        )

    # 4. context management
    middlewares.append(
        context_editing.build(
            trigger_tokens=context_edit_trigger_tokens,
            keep_tool_results=context_edit_keep_tool_results,
        )  # type: ignore
    )
    middlewares.append(
        summorization.build(
            model=model,
            trigger_tokens=summarization_trigger_tokens,
            keep_messages=summarization_keep_messages,
        )  # type: ignore
    )

    # 5. model fault tolerance (fallback wraps retry)
    fallback = model_fallback.build(fallback_models=fallback_models)
    if fallback is not None:
        middlewares.append(fallback)
    middlewares.append(
        model_retry.build(
            max_retries=model_max_retries,
            backoff_factor=retry_backoff_factor,
            initial_delay=retry_initial_delay,
            raise_on_failure=bool(fallback_models),
        )
    )

    # 6. tool fault tolerance (error handling wraps retry)
    middlewares.append(tool_error.build())
    middlewares.append(
        tool_retry.build(
            max_retries=tool_max_retries,
            backoff_factor=retry_backoff_factor,
            initial_delay=retry_initial_delay,
        )
    )

    subagent.keep_private_state(middlewares)

    return middlewares


__all__ = ["build_middlewares"]
