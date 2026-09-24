from deepagents.backends import FilesystemBackend
from deepagents.middleware.filesystem import FilesystemMiddleware

READ_ONLY_TOOLS = ["ls", "read_file", "glob", "grep"]
WRITE_TOOLS = ["write_file", "edit_file", "delete"]


def build(*, workspace_root: str, writable: bool) -> FilesystemMiddleware:
    """Expose the workspace to the agent as file tools.

    `workspace_root` arrives as an already-resolved absolute path string: this
    layer holds agent logic only, so normalizing paths is the caller's job.

    `FilesystemBackend` reads the real disk (unlike the default `StateBackend`,
    which is an in-state scratch filesystem). Write tools stay off unless
    `writable` is set.
    """
    tools = list(READ_ONLY_TOOLS)
    if writable:
        tools += WRITE_TOOLS

    return FilesystemMiddleware(
        backend=FilesystemBackend(root_dir=workspace_root),
        system_prompt=(
            "You have read access to the project workspace. "
            "Locate code with `glob`/`grep` before reading whole files."
        ),
        tools=tools,  # type: ignore[arg-type]
    )
