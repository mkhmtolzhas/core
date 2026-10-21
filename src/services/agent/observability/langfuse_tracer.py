import asyncio
import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langfuse import LangfuseAgent, propagate_attributes
from langfuse.langchain import CallbackHandler

from src.core.connections.observability.langfuse_connection import LangfuseConnection
from src.services.agent.observability.tracer import PromptTrace, Tracer

TRACE_NAME = "answer-prompt"
"""Name of the trace one prompt produces.

A stable name, not a per-run one: dashboards, evaluators and saved filters
select traces by it, and anything that varies between runs belongs in metadata.
"""

INTERRUPTED = "Interrupted by the user."

_MIDDLEWARE_HOOK = re.compile(r"^\w+Middleware\.(?:before|after)_(?:agent|model)$")
"""Run names `create_agent` gives the graph nodes of middleware hooks."""

_VISIBLE = object()


class _AgentCallbackHandler(CallbackHandler):
    """The LangChain integration, minus the agent's own bookkeeping.

    Every middleware hook is a graph node, so each model step arrives wrapped
    in four or five spans — call limits checked, todos synced — that say
    nothing about what the agent did and bury the steps that do. Worse, a hook
    whose name contains "agent" is typed as an agent and shows up as one in
    the Agent Graph. They are dropped, and anything they start themselves (the
    summarization middleware calls the model) is attached to the nearest
    observation that was kept.

    Overrides private hooks of the SDK's handler; re-check them on a Langfuse
    major upgrade.
    """

    def __init__(self, *, public_key: str) -> None:
        super().__init__(public_key=public_key)
        self._hidden: dict[UUID, Optional[UUID]] = {}

    def on_chain_start(
        self,
        serialized: Optional[dict[str, Any]],
        inputs: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        name = self.get_langchain_run_name(serialized, **kwargs)
        if parent_run_id is not None and _MIDDLEWARE_HOOK.match(name):
            # Still tracked, so the trace's cleanup at the root covers it.
            self._track_run(run_id=run_id, parent_run_id=parent_run_id, metadata=metadata)
            self._hidden[run_id] = parent_run_id
            return None

        return super().on_chain_start(
            serialized,
            inputs,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            **kwargs,
        )

    def on_chain_end(self, outputs: dict[str, Any], *, run_id: UUID, **kwargs: Any) -> Any:
        if self._hidden.pop(run_id, _VISIBLE) is not _VISIBLE:
            return None
        return super().on_chain_end(outputs, run_id=run_id, **kwargs)

    def on_chain_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        if self._hidden.pop(run_id, _VISIBLE) is not _VISIBLE:
            return None
        return super().on_chain_error(error, run_id=run_id, **kwargs)

    def _get_parent_observation(self, parent_run_id: Optional[UUID]) -> Any:
        while parent_run_id in self._hidden:
            parent_run_id = self._hidden[parent_run_id]
        return super()._get_parent_observation(parent_run_id)

    def _get_error_level_and_status_message(self, error: BaseException) -> Any:
        """A prompt the user aborted is not a failure of the steps it cut short.

        Without this every open span reports an error whose message is the
        repr of asyncio's cancellation sentinel.
        """
        if isinstance(error, asyncio.CancelledError):
            return "WARNING", INTERRUPTED
        return super()._get_error_level_and_status_message(error)


class _LangfusePromptTrace(PromptTrace):
    def __init__(self, observation: LangfuseAgent) -> None:
        self._observation = observation

    def record_answer(self, body: str) -> None:
        self._observation.update(output=body)


class LangfuseTracer(Tracer):
    """Traces every prompt to Langfuse.

    The LangChain integration does the real work: each model call becomes a
    generation with its model and token usage, each tool call a tool, each
    subagent its own nested run. What this adds is the root above them. Without
    it the trace would be named after the graph and carry its whole state as
    input and output — every message of the thread, since the checkpointer
    replays it — instead of the one prompt and the one reply.

    Takes the connection rather than a client, like the memories take theirs:
    the client exists only once the session has opened it.
    """

    def __init__(
        self, *, connection: LangfuseConnection, metadata: Mapping[str, str]
    ) -> None:
        self._connection = connection
        self._metadata = dict(metadata)

    @contextmanager
    def trace_prompt(self, *, session_id: str, prompt: str) -> Iterator[PromptTrace]:
        """Open the prompt's root observation and tag everything beneath it.

        `session_id` is the LangGraph thread, so the Sessions view groups
        prompts exactly as the checkpointer does — a resumed conversation
        continues the same session. Propagation stamps it on every child
        observation too, which is what lets session cost include the
        generations.

        An aborted prompt is marked rather than left looking like a success
        with a short answer: interruptions are worth filtering for on their own.
        """
        with (
            self._connection.connection_factory().start_as_current_observation(
                as_type="agent", name=TRACE_NAME, input=prompt
            ) as observation,
            propagate_attributes(
                session_id=session_id,
                trace_name=TRACE_NAME,
                metadata=self._metadata,
            ),
        ):
            try:
                yield _LangfusePromptTrace(observation)
            except asyncio.CancelledError:
                observation.update(level="WARNING", status_message=INTERRUPTED)
                raise
            except Exception as error:
                observation.update(
                    level="ERROR", status_message=f"{type(error).__name__}: {error}"
                )
                raise

    def callbacks(self) -> Sequence[BaseCallbackHandler]:
        return (_AgentCallbackHandler(public_key=self._connection.public_key),)
