from collections.abc import Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.messages import SystemMessage

from src.services.agent.types import AgentState, Context


class ContextMiddleware(AgentMiddleware[AgentState, Context]):
    """Set the base system prompt from the run's context.

    The service resolves the prompt once per turn and hands it in as
    `Context.system_prompt`, so this middleware depends on nothing: where the
    text comes from is the service's business.

    Must be the outermost middleware that touches the system message. The
    others — todos, filesystem, subagents — append their own instructions to
    whatever they receive, so anything placed outside this one would have its
    additions replaced rather than kept.
    """

    def wrap_model_call(
        self,
        request: ModelRequest[Context],
        handler: Callable[[ModelRequest[Context]], ModelResponse],
    ) -> ModelResponse:
        return handler(self._with_system_prompt(request))

    async def awrap_model_call(
        self,
        request: ModelRequest[Context],
        handler: Callable[[ModelRequest[Context]], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        return await handler(self._with_system_prompt(request))

    @staticmethod
    def _with_system_prompt(request: ModelRequest[Context]) -> ModelRequest[Context]:
        system_message = SystemMessage(content=request.runtime.context.system_prompt)
        return request.override(system_message=system_message)


def build() -> ContextMiddleware:
    return ContextMiddleware()
