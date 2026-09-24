"""Rich console factory."""

from rich.console import Console

from src.handler.cli.theme import RICH_THEME


def build_console() -> Console:
    """Console for everything printed outside the input box.

    Highlighting is off because the agent's answers are rendered as Markdown;
    Rich's automatic highlighter would otherwise recolour numbers and paths
    inside prose.
    """
    return Console(theme=RICH_THEME, highlight=False)
