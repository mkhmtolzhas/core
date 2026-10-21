import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from functools import lru_cache
from typing import Any, cast

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer, StreamMode

from src.core.config import settings
from src.entities.event import AgentEvent, TextChunk, ToolCall, ToolResult
from src.entities.prompt import Prompt
from src.services.agent.memory.long_term import get_long_term_memory
from src.services.agent.memory.short_term import get_short_term_memory
from src.services.agent.middlewares import build_middlewares
from src.services.agent.observability import Tracer, get_tracer
from src.services.agent.prompts import PromptSource, get_system_prompt_source
from src.services.agent.providers import build_model
from src.services.agent.tools import get_tools
from src.services.agent.types import AgentState, Context

UPDATES_MODE: StreamMode = "updates"
MESSAGES_MODE: StreamMode = "messages"

MODEL_NODE = "model"
TOOLS_NODE = "tools"
"""The two nodes worth showing.

Every middleware also reports its own steps under names like
`SummarizationMiddleware.before_model`, and those are bookkeeping, not work the
user asked for.
"""


def _events_of(node: str, update: Any) -> Sequence[AgentEvent]:
    """Translate one graph step into events, dropping everything else.

    The model node also reports the finished answer here, which the token
    stream has already delivered; only its tool calls are new.
    """
    if node not in (MODEL_NODE, TOOLS_NODE):
        return ()

    events: list[AgentEvent] = []
    for message in (update or {}).get("messages", []):
        if isinstance(message, AIMessage):
            events.extend(
                ToolCall(name=call["name"], arguments=call["args"])
                for call in message.tool_calls
            )
        elif isinstance(message, ToolMessage):
            events.append(ToolResult(name=message.name or "", body=message.text))
    return events


class AgentService(ABC):
    @abstractmethod
    def stream_answer(
        self, prompt: Prompt, thread_id: str
    ) -> AsyncIterator[AgentEvent]:
        """Answer a prompt as a sequence of events, in the order they happen.

        The thread is an argument rather than a property of the service: one
        graph, one checkpointer, and as many conversations as the caller has
        ids for. Switching sessions is then a different value here, not a
        rebuilt agent — which would reload the model to change a string.
        """
        raise NotImplementedError()


class AgentServiceImpl(AgentService):
    def __init__(
        self,
        llm: BaseChatModel,
        tools: Sequence[BaseTool],
        middlewares: Sequence[AgentMiddleware],
        system_prompt: PromptSource,
        tracer: Tracer,
        checkpointer: Checkpointer | None = None,
        store: BaseStore | None = None,
        state_schema: Any | None = None,
    ) -> None:
        """The system prompt is not baked into the graph.

        It is resolved on every turn and handed to the run as its `Context`,
        which `ContextMiddleware` from `middlewares` turns into the system
        message — so the chain passed in must include it.
        """
        self.llm = llm
        self.tracer = tracer
        self.system_prompt = system_prompt
        self.agent = create_agent(
            model=self.llm,
            tools=list(tools),
            middleware=list(middlewares),  # type: ignore
            checkpointer=checkpointer,
            store=store,
            state_schema=state_schema,
            context_schema=Context,
        )  # type: ignore

    async def stream_answer(
        self, prompt: Prompt, thread_id: str
    ) -> AsyncIterator[AgentEvent]:
        """Answer one prompt as the next turn of an ongoing conversation.

        Only the new message is sent: the checkpointer replays the thread
        identified by `thread_id`, so the agent still sees everything said
        before — including a thread from a previous run of the CLI.

        Both stream modes are consumed at once, and each kind of event is taken
        from exactly one of them. Text comes from `messages`, because that is
        the only mode that emits it before the turn is finished; tools come from
        `updates`, because a call is announced there before it runs. A tool
        result appears in both, and taking it twice would print it twice.

        The run itself happens in a task of its own, and this generator only
        relays what that task produces. Tracing is the reason: the open trace
        lives in context variables, and a caller that advances the stream from
        a fresh task on every step — `interruptible` does — would give each
        step its own copy of the context. Later observations would then lose
        the session id, and the root could not be closed where it was opened.
        """
        events: asyncio.Queue[AgentEvent | None] = asyncio.Queue()
        run = asyncio.create_task(self._run(prompt, thread_id, events))
        run.add_done_callback(lambda _: events.put_nowait(None))
        try:
            while (event := await events.get()) is not None:
                yield event
            run.result()  # the run's own failure, if it had one
        finally:
            if not run.done():
                run.cancel()
                await asyncio.gather(run, return_exceptions=True)

    async def _run(
        self, prompt: Prompt, thread_id: str, events: asyncio.Queue[AgentEvent | None]
    ) -> None:
        """Answer the prompt inside its trace, start to finish in this task.

        The reply recorded is the text after the last tool event: what the
        model said before calling a tool was a step on the way, not the answer.

        The system prompt is fetched here, once, so a turn never switches
        versions halfway through its steps.
        """
        with self.tracer.trace_prompt(
            session_id=thread_id, prompt=prompt.body
        ) as trace:
            context = Context(system_prompt=await self.system_prompt.get())
            answer: list[str] = []
            try:
                async for event in self._stream(prompt, thread_id, context):
                    if isinstance(event, TextChunk):
                        answer.append(event.body)
                    else:
                        answer.clear()
                    events.put_nowait(event)
            finally:
                trace.record_answer("".join(answer))

    async def _stream(
        self, prompt: Prompt, thread_id: str, context: Context
    ) -> AsyncIterator[AgentEvent]:
        stream = self.agent.astream(
            {"messages": [{"role": "user", "content": prompt.body}]},
            config={
                "configurable": {"thread_id": thread_id},
                "callbacks": list(self.tracer.callbacks()),
            },
            context=context,
            stream_mode=[UPDATES_MODE, MESSAGES_MODE],
        )
        async for payload in stream:
            mode, chunk = cast(tuple[str, Any], payload)
            if mode == MESSAGES_MODE:
                token, metadata = chunk
                if metadata.get("langgraph_node") != MODEL_NODE:
                    continue
                body = getattr(token, "text", "")
                if body:
                    yield TextChunk(body=body)
                continue

            for node, update in chunk.items():
                for event in _events_of(node, update):
                    yield event


@lru_cache(maxsize=1)
def get_agent_service() -> AgentService:
    """Composition root: the only place that reads `settings`.

    Everything below receives plain values, so no component depends on the
    configuration object itself.

    Must be called from inside a running event loop and after the database
    connection is open: the store binds itself to the loop it is built on. The
    lazy resolution in the layers above already guarantees both.

    One service serves every session: which conversation a prompt belongs to is
    decided by the thread id passed to `stream`, and that id is owned by the
    use case, which is also what records the session it belongs to.
    """
    llm = build_model(
        provider=settings.provider,
        model=settings.model,
        api_key=settings.api_key,
        num_ctx=settings.num_ctx,
    )
    tools = get_tools()

    return AgentServiceImpl(
        llm=llm,
        checkpointer=get_short_term_memory().create_short_term_memory(),
        store=get_long_term_memory().create_long_term_memory(),
        tools=tools,
        middlewares=build_middlewares(
            model=llm,
            workspace_root=str(settings.workspace_root),
            thread_call_limit=settings.thread_call_limit,
            run_call_limit=settings.run_call_limit,
            context_edit_trigger_tokens=settings.context_edit_trigger_tokens,
            context_edit_keep_tool_results=settings.context_edit_keep_tool_results,
            summarization_trigger_tokens=settings.summarization_trigger_tokens,
            summarization_keep_messages=settings.summarization_keep_messages,
            fallback_models=settings.fallback_models,
            model_max_retries=settings.model_max_retries,
            tool_max_retries=settings.tool_max_retries,
            retry_backoff_factor=settings.retry_backoff_factor,
            retry_initial_delay=settings.retry_initial_delay,
            enable_todos=settings.enable_todos,
            enable_filesystem=settings.enable_filesystem,
            filesystem_writable=settings.filesystem_writable,
            enable_file_search=settings.enable_file_search,
            enable_shell=settings.enable_shell,
            enable_subagents=settings.enable_subagents,
            tools=tools,
        ),
        system_prompt=get_system_prompt_source(),
        tracer=get_tracer(),
        state_schema=AgentState,
    )
