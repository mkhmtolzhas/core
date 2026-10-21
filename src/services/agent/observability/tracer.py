from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager, ExitStack, contextmanager

from langchain_core.callbacks import BaseCallbackHandler


class PromptTrace(ABC):
    """The trace one prompt is recorded under, while it is being answered."""

    @abstractmethod
    def record_answer(self, body: str) -> None:
        """Record the reply the user was given as the trace's output."""
        raise NotImplementedError()


class Tracer(ABC):
    """How an agent run is traced, if at all."""

    @abstractmethod
    def trace_prompt(
        self, *, session_id: str, prompt: str
    ) -> AbstractContextManager[PromptTrace]:
        """Open the trace everything one prompt causes is nested under.

        Must be entered and left in the same task: the open trace is held in a
        context variable, and another task would see a copy taken before it
        existed.
        """
        raise NotImplementedError()

    @abstractmethod
    def callbacks(self) -> Sequence[BaseCallbackHandler]:
        """Fresh LangChain callbacks for one run.

        Fresh because a handler keeps per-run state, and two runs sharing one
        would read each other's trace ids.
        """
        raise NotImplementedError()


class _NoopPromptTrace(PromptTrace):
    def record_answer(self, body: str) -> None:
        pass


class NoopTracer(Tracer):
    """Records nothing. Used when Langfuse is not configured."""

    @contextmanager
    def trace_prompt(self, *, session_id: str, prompt: str) -> Iterator[PromptTrace]:
        yield _NoopPromptTrace()

    def callbacks(self) -> Sequence[BaseCallbackHandler]:
        return ()


class _FanOutPromptTrace(PromptTrace):
    def __init__(self, traces: Sequence[PromptTrace]) -> None:
        self._traces = traces

    def record_answer(self, body: str) -> None:
        for trace in self._traces:
            trace.record_answer(body)


class FanOutTracer(Tracer):
    """Traces every prompt to several backends at once.

    Each backend opens its own trace and gets its own callbacks; none of them
    knows the others exist. Traces are entered in order and left in reverse,
    so an interrupted or failed prompt reaches every backend the same way.
    """

    def __init__(self, tracers: Sequence[Tracer]) -> None:
        self._tracers = tuple(tracers)

    @contextmanager
    def trace_prompt(self, *, session_id: str, prompt: str) -> Iterator[PromptTrace]:
        with ExitStack() as stack:
            yield _FanOutPromptTrace(
                [
                    stack.enter_context(
                        tracer.trace_prompt(session_id=session_id, prompt=prompt)
                    )
                    for tracer in self._tracers
                ]
            )

    def callbacks(self) -> Sequence[BaseCallbackHandler]:
        return tuple(
            callback for tracer in self._tracers for callback in tracer.callbacks()
        )
