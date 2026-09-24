"""Filesystem locations owned by the terminal handler.

This is the only module of the handler that touches `pathlib`; everything
above it receives plain strings, so no path object leaks toward the use-case
or the agent service.

Where per-user state lives is not decided here: that is configuration, and it
is owned by `settings.state_root` so the database and the history file cannot
drift apart.
"""

from pathlib import Path

from src.core.config import settings


def get_history_path() -> str:
    """Absolute path of the input-history file."""
    return str(settings.history_path())


def get_workspace() -> str:
    """Directory the session was started from, shown in the banner."""
    return str(Path.cwd())
