"""Langfuse connection.

Owns the SDK client and nothing else: which runs get traced and what a trace
looks like is the agent service's business (`src/services/agent/observability`).
"""

import asyncio
import logging
from typing import Optional

from langfuse import Langfuse

from src.core.config import settings
from src.core.connections.observability.masking import mask_secrets
from src.core.connections.observability.observability_connection import (
    ObservabilityConnection,
)
from src.core.loggers.file_logger import file_logger

SDK_LOGGER = "langfuse"


class LangfuseConnection(ObservabilityConnection[Langfuse]):
    """The Langfuse client, open for the length of the session.

    A connection because the SDK buffers: spans are exported from a background
    thread in batches, so whatever the last prompt produced is still in memory
    when the session ends, and only `close` sends it.

    Disabled — no client at all — unless both project keys are set. Not a
    disabled client: that one still complains about the missing key on stderr,
    which here is the terminal the user is typing into.
    """

    def __init__(
        self,
        *,
        public_key: Optional[str],
        secret_key: Optional[str],
        base_url: str,
        environment: str,
        release: Optional[str],
        sample_rate: float,
    ) -> None:
        self._public_key = public_key
        self._secret_key = secret_key
        self._base_url = base_url
        self._environment = environment
        self._release = release
        self._sample_rate = sample_rate
        self._client: Optional[Langfuse] = None

    @property
    def enabled(self) -> bool:
        return bool(self._public_key and self._secret_key)

    @property
    def public_key(self) -> str:
        """Which project the client sends to; the LangChain handler looks it up by this."""
        if not self._public_key:
            raise RuntimeError("Langfuse is not configured.")
        return self._public_key

    async def connect(self) -> None:
        """Build the client. Nothing is sent until the first trace ends.

        A slow or unreachable Langfuse therefore costs a log line, never a
        delayed answer.
        """
        if not self.enabled or self._client is not None:
            return

        _log_sdk_to_file()
        self._client = Langfuse(
            public_key=self._public_key,
            secret_key=self._secret_key,
            base_url=self._base_url,
            environment=self._environment,
            release=self._release,
            sample_rate=self._sample_rate,
            mask=mask_secrets,
        )

    async def close(self) -> None:
        """Send what is still buffered, then stop the exporter.

        Blocking network I/O, so it runs off the event loop.
        """
        if self._client is None:
            return

        await asyncio.to_thread(self._client.shutdown)
        self._client = None

    def connection_factory(self) -> Langfuse:
        if self._client is None:
            raise RuntimeError("Langfuse connection is not initialized.")
        return self._client


def _log_sdk_to_file() -> None:
    """Send the SDK's own warnings to the log file instead of the terminal.

    The SDK's logger has no handler, so Python prints its warnings to stderr —
    into the middle of the prompt or a streaming answer. A prompt that is not
    yet created in Langfuse would do that on every turn.
    """
    sdk_logger = logging.getLogger(SDK_LOGGER)
    sdk_logger.propagate = False
    for handler in file_logger.logger.handlers:
        if handler not in sdk_logger.handlers:
            sdk_logger.addHandler(handler)


langfuse_connection = LangfuseConnection(
    public_key=settings.langfuse_public_key,
    secret_key=settings.langfuse_secret_key,
    base_url=settings.langfuse_base_url,
    environment=settings.langfuse_tracing_environment,
    release=settings.langfuse_release,
    sample_rate=settings.langfuse_sample_rate,
)
