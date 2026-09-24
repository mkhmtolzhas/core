from langchain.agents import AgentState as BaseAgentState
from pydantic import BaseModel, ConfigDict


class AgentServiceModel(BaseModel):
    """Base model for AI service structured output."""

    model_config = ConfigDict(extra="ignore", use_enum_values=False)


class AgentState(BaseAgentState):
    """Custom Agent State to hold additional information."""

    pass


class Context:
    pass
