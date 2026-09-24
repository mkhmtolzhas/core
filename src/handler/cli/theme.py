"""Visual identity of the CORE terminal.

Colours live in one place so the Rich renderer and the prompt-toolkit input
box stay in sync: Rich styles are looked up by name through `RICH_THEME`,
prompt-toolkit ones through `PROMPT_STYLE`.
"""

from prompt_toolkit.styles import Style
from rich.theme import Theme

ACCENT = "#d97757"
"""Brand colour: logo, the agent marker, the input caret."""

TEXT = "#d0cec9"
MUTED = "#8a8781"
FAINT = "#4a4845"
DANGER = "#e0645c"
SUCCESS = "#7fb069"

RICH_THEME = Theme(
    {
        "core.accent": ACCENT,
        "core.text": TEXT,
        "core.muted": MUTED,
        "core.faint": FAINT,
        "core.danger": DANGER,
        "core.success": SUCCESS,
        "core.command": f"bold {ACCENT}",
        "core.tool": MUTED,
        "core.tool.name": f"bold {MUTED}",
        "core.tool.result": FAINT,
        # Rich's Markdown defaults paint code on a hard black background, which
        # fights the terminal's own. Flatten it onto the CORE palette instead.
        "markdown.code": ACCENT,
        "markdown.code_block": TEXT,
        "markdown.h1": f"bold {ACCENT}",
        "markdown.h2": f"bold {ACCENT}",
        "markdown.h3": f"bold {TEXT}",
        "markdown.item.bullet": ACCENT,
        "markdown.item.number": ACCENT,
        "markdown.block_quote": MUTED,
        "markdown.hr": FAINT,
        "markdown.link": ACCENT,
    }
)

CODE_THEME = "ansi_dark"
"""Pygments theme for fenced code: ANSI colours, so it keeps the terminal background."""

PROMPT_STYLE = Style.from_dict(
    {
        "frame.border": FAINT,
        "prompt.marker": ACCENT,
        "prompt.continuation": FAINT,
        "hint": MUTED,
        # Completion popup for slash commands.
        "completion-menu.completion": f"bg:#2a2724 {TEXT}",
        "completion-menu.completion.current": f"bg:{ACCENT} #1a1816",
        "completion-menu.meta.completion": f"bg:#2a2724 {MUTED}",
        "completion-menu.meta.completion.current": f"bg:{ACCENT} #1a1816",
    }
)

SPINNER_FRAMES = "dots"
"""Name of the Rich spinner used while the agent is working."""

REFRESH_RATE = 12
"""Live redraws per second while the answer is streaming in."""
