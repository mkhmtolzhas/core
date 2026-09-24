from langchain.agents.middleware import FilesystemFileSearchMiddleware


def build(*, workspace_root: str) -> FilesystemFileSearchMiddleware:
    """Standalone `glob_search`/`grep_search` over the real workspace.

    Only worth enabling when `FilesystemMiddleware` runs on a state backend;
    alongside the disk-backed filesystem tools it duplicates `glob`/`grep` and
    just gives a small local model two ways to do the same thing.
    """
    return FilesystemFileSearchMiddleware(
        root_path=workspace_root,
        use_ripgrep=True,
    )
