from langchain.agents.middleware import HostExecutionPolicy, ShellToolMiddleware


def build(*, workspace_root: str) -> ShellToolMiddleware:
    """Persistent shell session rooted at the workspace.

    `HostExecutionPolicy` runs commands directly on this machine with no
    sandbox; swap in `DockerExecutionPolicy` for untrusted work.
    """
    return ShellToolMiddleware(
        workspace_root=workspace_root,
        execution_policy=HostExecutionPolicy(),
    )
