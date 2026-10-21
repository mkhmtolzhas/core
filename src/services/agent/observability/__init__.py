"""Tracing agent runs to observability backends.

One tracer per enabled backend, each built on that backend's connection from
`src/core/connections/observability`. None enabled is a no-op, several are
fanned out: tracing is something the agent reports to, never something it
depends on, so a missing key or a backend that is down must not change how a
prompt is answered.

Adding a backend is two pieces: an `ObservabilityConnection` in core that owns
its client, and a `Tracer` here that turns a run into its traces — then one
more branch below.
"""

from functools import lru_cache
from typing import List

from src.core.config import settings
from src.core.connections.observability.langfuse_connection import (
    langfuse_connection,
)
from src.services.agent.observability.langfuse_tracer import LangfuseTracer
from src.services.agent.observability.tracer import (
    FanOutTracer,
    NoopTracer,
    PromptTrace,
    Tracer,
)


@lru_cache(maxsize=1)
def get_tracer() -> Tracer:
    metadata = {"provider": settings.provider.value, "model": settings.model}

    tracers: List[Tracer] = []
    if langfuse_connection.enabled:
        tracers.append(LangfuseTracer(connection=langfuse_connection, metadata=metadata))

    if not tracers:
        return NoopTracer()
    if len(tracers) == 1:
        return tracers[0]
    return FanOutTracer(tracers)


__all__ = ["get_tracer", "PromptTrace", "Tracer"]
