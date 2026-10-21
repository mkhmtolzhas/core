import asyncio

from src.core.connections.observability.langfuse_connection import LangfuseConnection
from src.core.loggers.logger import Logger
from src.services.agent.prompts.source import PromptSource

FETCH_TIMEOUT_SECONDS = 2
"""How long a fetch with nothing in the cache may wait for Langfuse.

The SDK default is 5 seconds per attempt over 3 attempts with a pause between
them, so an unreachable Langfuse would hold the first prompt for about
17 seconds before falling back. One short attempt instead: the fallback is a
working prompt, and a failed fetch is retried on the next turn anyway.
"""


class LangfusePromptSource(PromptSource):
    """A prompt managed in Langfuse, served from the SDK's cache.

    `get_prompt` is synchronous but rarely touches the network: the SDK keeps
    every prompt in memory for 60 seconds and, once that expires, returns the
    stale version immediately while refreshing it in a background thread. Only
    a fetch with nothing cached blocks, which is why it runs off the event loop.

    Takes the connection rather than a client, like the tracer does: the client
    exists only once the session has opened it.
    """

    def __init__(
        self,
        *,
        connection: LangfuseConnection,
        name: str,
        label: str,
        fallback: str,
        logger: Logger,
    ) -> None:
        self._connection = connection
        self._name = name
        self._label = label
        self._fallback = fallback
        self._logger = logger

    async def get(self) -> str:
        """The `label` version of the prompt, or the fallback if there is none.

        A prompt not yet created in Langfuse ends up here too, so the agent
        works the same before anyone has set the prompt up.
        """
        try:
            return await asyncio.to_thread(self._fetch)
        except Exception as error:
            self._logger.warning(
                f"Prompt '{self._name}' ({self._label}) unavailable, using the "
                f"built-in one. {type(error).__name__}: {error}"
            )
            return self._fallback

    def _fetch(self) -> str:
        prompt = self._connection.connection_factory().get_prompt(
            self._name,
            label=self._label,
            type="text",
            max_retries=0,
            fetch_timeout_seconds=FETCH_TIMEOUT_SECONDS,
        )
        return prompt.compile()
