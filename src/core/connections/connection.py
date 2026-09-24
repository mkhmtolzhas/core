from abc import ABC, abstractmethod


class Connection(ABC):
    @abstractmethod
    async def connect(self):
        """Establish a connection."""
        raise NotImplementedError()

    @abstractmethod
    async def close(self):
        """Close the connection."""
        raise NotImplementedError()
