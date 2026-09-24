from typing import Optional

from langchain.agents.middleware import ToolCallRequest, ToolErrorMiddleware

RECOVERABLE = (ValueError, TypeError, KeyError, FileNotFoundError, TimeoutError)


def _explain(exc: Exception, request: ToolCallRequest) -> Optional[str]:
    """Turn a recoverable tool failure into feedback the model can act on.

    Only the exception type is surfaced — raw messages can carry internal or
    sensitive detail. Returning `None` lets everything else propagate and halt
    the run rather than being silently serialized to the model.
    """
    if isinstance(exc, RECOVERABLE):
        return (
            f"`{request.tool_call['name']}` failed with {type(exc).__name__}. "
            "Check the arguments and try a different approach."
        )
    return None


async def _aexplain(exc: Exception, request: ToolCallRequest) -> Optional[str]:
    return _explain(exc, request)


def build() -> ToolErrorMiddleware:
    """Outermost tool wrapper: converts exhausted failures into `ToolMessage`s."""
    return ToolErrorMiddleware(on_error=_explain, aon_error=_aexplain)
