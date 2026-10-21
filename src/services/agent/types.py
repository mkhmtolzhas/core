from dataclasses import dataclass

from langchain.agents import AgentState as BaseAgentState
from pydantic import BaseModel, ConfigDict


class AgentServiceModel(BaseModel):
    """Base model for AI service structured output."""

    model_config = ConfigDict(extra="ignore", use_enum_values=False)


class AgentState(BaseAgentState):
    """Custom Agent State to hold additional information."""

    pass


@dataclass(frozen=True)
class Context:
    """What one turn hands to every step of its run."""

    system_prompt: str
    """Base system prompt, resolved once per turn.

    Once rather than per model call, so every step of a turn runs on the same
    version even if the prompt is edited in Langfuse halfway through it.
    """
