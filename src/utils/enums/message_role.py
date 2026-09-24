"""Who — or what — produced one recorded turn of a conversation."""

from enum import Enum


class MessageRole(str, Enum):
    """The four kinds of line a session transcript can hold.

    A tool call and its result are separate roles rather than one record: the
    call is decided by the model and the result is produced by the tool, they
    are minutes apart when the tool is slow, and a run can be interrupted
    between them. Storing them as one row would have to invent a result that
    never arrived.
    """

    USER = "user"
    ASSISTANT = "assistant"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
