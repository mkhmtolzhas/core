from typing import List, Sequence

from deepagents.backends import FilesystemBackend
from deepagents.middleware._state import private_state_field_names
from deepagents.middleware.subagents import (
    GENERAL_PURPOSE_SUBAGENT,
    SubAgent,
    SubAgentMiddleware,
)
from langchain.agents.middleware import AgentMiddleware
from langchain.tools import BaseTool
from langchain_core.language_models import BaseChatModel

from src.services.agent.middlewares.filesystem import build as build_filesystem
from src.services.agent.middlewares.model_call_limit import build as build_call_limit

RESEARCHER_NAME = "researcher"

RESEARCHER_DESCRIPTION = (
    "Read-only investigator. Give it a question about the codebase and it "
    "reads whatever it needs and comes back with an answer. Prefer it over "
    "reading files yourself whenever finding the answer will take more than a "
    "couple of lookups."
)

RESEARCHER_PROMPT = (
    "You investigate a codebase and report what you found. The workspace is "
    "mounted at `/`, so `/src` is the project's src directory.\n"
    "Work by reading, not by guessing. Locate code with `glob`/`grep`, then "
    "open the files: a name tells you where to look, never what is inside. "
    "In particular, do not infer the framework or architecture from filenames "
    "— read the imports.\n"
    "Before concluding that something does not exist, open the `__init__.py` "
    "of every package that could plausibly contain it, and search for the "
    "concept rather than for one spelling of it: the thing you are asked about "
    "may be assembled by a function whose name you did not think to grep for.\n"
    "Never turn `I did not find it` into `it is not there`. Say which paths and "
    "patterns you checked, and state plainly that the search was inconclusive "
    "when it was — a wrong confident answer is worse than an admitted gap, "
    "because the agent that called you cannot see your work and will repeat "
    "whatever you claim.\n"
    "You cannot modify anything and must not suggest that you did. The agent "
    "that called you sees only your final message, so quote the lines that "
    "matter and name the paths they came from."
)


def _own_middlewares(
    *,
    workspace_root: str,
    thread_call_limit: int,
    run_call_limit: int,
) -> List[AgentMiddleware]:
    """The chain every subagent gets, built fresh for each of them.

    A middleware instance carries per-agent state, so the two roles cannot share
    one: they would then share a call budget, and the second subagent would
    start out with the first one's spending already counted against it.
    """
    return [
        build_call_limit(
            thread_limit=thread_call_limit,
            run_limit=run_call_limit,
        ),  # type: ignore[list-item]
        build_filesystem(
            workspace_root=workspace_root,
            writable=False,
        ),  # type: ignore[list-item]
    ]


def build(
    *,
    model: BaseChatModel,
    workspace_root: str,
    tools: Sequence[BaseTool],
    thread_call_limit: int,
    run_call_limit: int,
) -> SubAgentMiddleware:
    """Let the agent hand a self-contained job to a fresh agent.

    The point on a small context window is isolation, not parallelism: a search
    that would otherwise fill the conversation with twenty tool results happens
    somewhere else, and only the conclusion comes back.

    Two roles rather than one. `general-purpose` carries the same tools as the
    parent and takes its character entirely from the task it is given, which is
    how the agent improvises a role that was never declared here. `researcher`
    is the case worth naming: it exists so that a read-only question cannot
    reach a write tool by accident, and so the model has an obvious choice for
    the thing it will want most often.

    Both get their own call limit. A subagent runs its own loop, and without
    one a confused nested agent would spin until the process is killed —
    the parent's budget does not reach inside it.
    """
    general: SubAgent = {
        **GENERAL_PURPOSE_SUBAGENT,  # type: ignore[typeddict-item]
        "model": model,
        "tools": list(tools),
        "middleware": _own_middlewares(
            workspace_root=workspace_root,
            thread_call_limit=thread_call_limit,
            run_call_limit=run_call_limit,
        ),
    }

    researcher: SubAgent = {
        "name": RESEARCHER_NAME,
        "description": RESEARCHER_DESCRIPTION,
        "system_prompt": RESEARCHER_PROMPT,
        "model": model,
        "tools": [],
        "middleware": _own_middlewares(
            workspace_root=workspace_root,
            thread_call_limit=thread_call_limit,
            run_call_limit=run_call_limit,
        ),
    }

    return SubAgentMiddleware(
        backend=FilesystemBackend(root_dir=workspace_root),
        subagents=[general, researcher],
    )


def keep_private_state(middlewares: Sequence[AgentMiddleware]) -> None:
    """Stop subagents from inheriting state the parent keeps to itself.

    A subagent starts from a copy of its parent's state, minus the keys
    `SubAgentMiddleware` is told are private — and it is told nothing unless
    someone collects them from the other middlewares' schemas.
    `create_deep_agent` does that; building the middleware by hand skips it.
    The shell middleware is the case that breaks: its live session sits in
    state, and once copied into a subagent the checkpointer tries to serialise
    it and every `task` call fails with `Type is not msgpack serializable`.

    Collected from the whole chain, so it must run once the chain is complete.
    """
    private = private_state_field_names(
        *(middleware.state_schema for middleware in middlewares)
    )
    for middleware in middlewares:
        if isinstance(middleware, SubAgentMiddleware):
            middleware.private_state_keys = middleware.private_state_keys | private
