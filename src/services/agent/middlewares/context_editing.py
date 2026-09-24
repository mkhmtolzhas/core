from langchain.agents.middleware import ClearToolUsesEdit, ContextEditingMiddleware


def build(*, trigger_tokens: int, keep_tool_results: int) -> ContextEditingMiddleware:
    """Drop stale tool outputs once the context grows past `trigger_tokens`.

    Deterministic and free, so it is placed before summarization: pruning old
    tool results often keeps the window small enough that no LLM-based
    compaction is needed at all.
    """
    return ContextEditingMiddleware(
        edits=[
            ClearToolUsesEdit(
                trigger=trigger_tokens,
                keep=keep_tool_results,
                clear_tool_inputs=False,
                exclude_tools=(),
                placeholder="[cleared]",
            ),
        ],
    )
