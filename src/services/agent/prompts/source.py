from abc import ABC, abstractmethod


class PromptSource(ABC):
    """Where the text of one prompt comes from."""

    @abstractmethod
    async def get(self) -> str:
        """Return the prompt's current text.

        Never raises: a prompt store that is down must not stop the agent from
        answering, so an implementation that can fail falls back to a text it
        was given instead.
        """
        raise NotImplementedError()


class StaticPromptSource(PromptSource):
    """A prompt fixed in code. Used when no prompt store is configured."""

    def __init__(self, text: str) -> None:
        self._text = text

    async def get(self) -> str:
        return self._text
