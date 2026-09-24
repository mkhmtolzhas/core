"""Answering a prompt, and keeping a record of having answered it.

The use case is where the two concerns meet: the agent service knows how to
produce events and nothing about storage, the repositories know how to store
rows and nothing about agents. Recording here also means the transcript is
written from the same event stream the user sees, so the two can never tell
different stories.

It is also what holds the *current* conversation. The thread id lives here
rather than in the agent service because a thread is a record, not a runtime:
the row in `sessions` and the checkpoints LangGraph writes are two halves of
the same thing, and moving between conversations must move both together.
"""

from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any, AsyncIterator, Callable, List, Mapping, Optional
from uuid import UUID, uuid4

from src.core.config import settings
from src.core.loggers.file_logger import file_logger
from src.core.loggers.logger import Logger
from src.entities.event import AgentEvent, TextChunk, ToolCall, ToolResult
from src.entities.message import Message
from src.entities.prompt import Prompt
from src.entities.session import Session, SessionDigest
from src.repos.message_repository import MessageRepository, get_message_repository
from src.repos.session_repository import SessionRepository, get_session_repository
from src.services.agent.service import AgentService, get_agent_service
from src.utils.enums.message_role import MessageRole
from src.utils.enums.provider import Provider


class AgentUseCase(ABC):
    @property
    @abstractmethod
    def current_session_id(self) -> Optional[UUID]:
        """Session the next prompt will be recorded under, once one exists.

        `None` until the first prompt of a fresh conversation: a session that
        answered nothing is not a session worth a row.
        """
        raise NotImplementedError()

    @abstractmethod
    def stream_answer(self, prompt: Prompt) -> AsyncIterator[AgentEvent]:
        """Answer a prompt as a sequence of events, in the order they happen."""
        raise NotImplementedError()

    @abstractmethod
    async def recent_sessions(self, limit: int = 20) -> List[SessionDigest]:
        """List the sessions that can be resumed, most recently used first."""
        raise NotImplementedError()

    @abstractmethod
    async def resume_session(self, session_id: UUID) -> List[Message]:
        """Continue an earlier session, and hand back everything said in it."""
        raise NotImplementedError()

    @abstractmethod
    def start_new_session(self) -> None:
        """Leave the current conversation and begin an empty one."""
        raise NotImplementedError()

    @abstractmethod
    async def session_history(self) -> List[Message]:
        """Return the current session's turns, oldest first."""
        raise NotImplementedError()


class AgentUseCaseImpl(AgentUseCase):
    def __init__(
        self,
        agent_service_factory: Callable[[], AgentService],
        session_repository: SessionRepository,
        message_repository: MessageRepository,
        logger: Logger,
        provider: Provider,
        model: str,
        workspace_root: str,
    ) -> None:
        self._agent_service_factory = agent_service_factory
        self._agent_service: Optional[AgentService] = None
        self._session_repository = session_repository
        self._message_repository = message_repository
        self._logger = logger
        self._provider = provider
        self._model = model
        self._workspace_root = workspace_root
        self._session_id: Optional[UUID] = None
        self._thread_id: str = str(uuid4())

    @property
    def agent_service(self) -> AgentService:
        """Built on first use, so nothing is loaded before it is needed."""
        if self._agent_service is None:
            self._agent_service = self._agent_service_factory()
        return self._agent_service

    @property
    def current_session_id(self) -> Optional[UUID]:
        return self._session_id

    async def _open_session(self) -> Optional[UUID]:
        """Create the session row on the first prompt, not when it is started.

        A conversation nobody spoke to leaves nothing behind, so `/new` costs a
        thread id and nothing else. The id is minted up front all the same: it
        is what the checkpointer files this turn under, and it has to be the
        same value the row is written with or the two could not be joined.
        """
        if self._session_id is not None:
            return self._session_id

        try:
            self._session_id = await self._session_repository.create(
                Session(
                    thread_id=self._thread_id,
                    provider=self._provider,
                    model=self._model,
                    workspace_root=self._workspace_root,
                )
            )
        except Exception as error:
            # Recording is bookkeeping. A database that will not take a row is
            # worth a line in the log, not a session the user cannot use.
            self._logger.error(f"Failed to open a session record: {error}")
            return None

        self._logger.info(
            f"Session opened on {self._provider.value}:{self._model} "
            f"(thread {self._thread_id})."
        )
        return self._session_id

    async def _record(
        self,
        session_id: UUID,
        role: MessageRole,
        body: str = "",
        tool_name: Optional[str] = None,
        tool_arguments: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Append one turn. Its position is the moment it was written."""
        try:
            await self._message_repository.create(
                Message(
                    session_id=session_id,
                    role=role,
                    body=body,
                    tool_name=tool_name,
                    tool_arguments=dict(tool_arguments) if tool_arguments else None,
                )
            )
        except Exception as error:
            self._logger.error(f"Failed to record a {role.value} message: {error}")

    async def stream_answer(self, prompt: Prompt) -> AsyncIterator[AgentEvent]:
        """Answer one prompt, writing down what happened as it happens.

        The answer arrives token by token but is stored as one message, so the
        text is buffered and flushed whenever a tool interrupts it and again
        when the turn ends. The flush lives in a `finally`: an interrupted run
        still produced the words it produced, and losing them would make the
        transcript disagree with the screen.
        """
        session_id = await self._open_session()
        thread_id = self._thread_id

        if session_id is None:
            async for event in self.agent_service.stream_answer(prompt, thread_id):
                yield event
            return

        await self._record(session_id, MessageRole.USER, body=prompt.body)

        answer: List[str] = []

        async def flush() -> None:
            if not answer:
                return
            body = "".join(answer)
            answer.clear()
            await self._record(session_id, MessageRole.ASSISTANT, body=body)

        try:
            async for event in self.agent_service.stream_answer(prompt, thread_id):
                if isinstance(event, TextChunk):
                    answer.append(event.body)
                elif isinstance(event, ToolCall):
                    await flush()
                    await self._record(
                        session_id,
                        MessageRole.TOOL_CALL,
                        tool_name=event.name,
                        tool_arguments=event.arguments,
                    )
                elif isinstance(event, ToolResult):
                    await flush()
                    await self._record(
                        session_id,
                        MessageRole.TOOL_RESULT,
                        body=event.body,
                        tool_name=event.name,
                    )
                yield event
        finally:
            await flush()

    async def recent_sessions(self, limit: int = 20) -> List[SessionDigest]:
        try:
            return await self._session_repository.recent(limit=limit)
        except Exception as error:
            self._logger.error(f"Failed to list sessions: {error}")
            return []

    async def resume_session(self, session_id: UUID) -> List[Message]:
        """Point both halves of the conversation at an earlier session.

        The thread id comes from the stored row, so the very next prompt is
        checkpointed onto the same LangGraph thread the agent wrote before —
        which is what makes the model remember the conversation rather than
        merely the user reading it back.
        """
        session = await self._session_repository.get(session_id)

        self._session_id = session.id
        self._thread_id = session.thread_id
        self._logger.info(f"Resumed session on thread {session.thread_id}.")

        return await self.session_history()

    def start_new_session(self) -> None:
        self._session_id = None
        self._thread_id = str(uuid4())
        self._logger.info(f"New session started on thread {self._thread_id}.")

    async def session_history(self) -> List[Message]:
        if self._session_id is None:
            return []
        try:
            return await self._message_repository.history(self._session_id)
        except Exception as error:
            self._logger.error(f"Failed to read the session transcript: {error}")
            return []


@lru_cache(maxsize=1)
def get_agent_use_case() -> AgentUseCase:
    """Composition root of the use-case layer."""
    return AgentUseCaseImpl(
        agent_service_factory=get_agent_service,
        session_repository=get_session_repository(),
        message_repository=get_message_repository(),
        logger=file_logger,
        provider=settings.provider,
        model=settings.model,
        workspace_root=str(settings.workspace_root),
    )
