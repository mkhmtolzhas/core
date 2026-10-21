from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from src.core.connections.connection import Connection

ClientT = TypeVar("ClientT")


class ObservabilityConnection(Connection, ABC, Generic[ClientT]):
    """A backend traces are sent to — Langfuse, LangSmith, an OTel collector.

    Optional by nature, unlike a database: the agent answers the same with or
    without one, so a backend that is not configured is simply off rather
    than an error, and several can be on at once.

    `ClientT` is the backend's own SDK client. It is what the matching tracer
    in the agent service needs, and nothing above that tracer should see it.
    """

    @property
    @abstractmethod
    def enabled(self) -> bool:
        """Whether the backend is configured.

        A disabled connection opens nothing on `connect` and is never handed
        to a tracer, so its `connection_factory` is never called.
        """
        raise NotImplementedError()

    @abstractmethod
    def connection_factory(self) -> ClientT:
        """Return the live client, once `connect` has opened it."""
        raise NotImplementedError()
