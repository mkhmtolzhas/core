"""Text logo for CORE.

A drawn logo is used when the terminal is wide enough for it, and a one-line
badge otherwise, so the banner never wraps into noise.
"""

from typing import Tuple

NAME = "CORE"
TAGLINE = "local engineering agent"

LOGO_LINES: Tuple[str, ...] = (
    " ██████╗ ██████╗ ██████╗ ███████╗",
    "██╔════╝██╔═══██╗██╔══██╗██╔════╝",
    "██║     ██║   ██║██████╔╝█████╗  ",
    "██║     ██║   ██║██╔══██╗██╔══╝  ",
    "╚██████╗╚██████╔╝██║  ██║███████╗",
    " ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝",
)

BADGE = f"◆ {NAME}"
"""Fallback for terminals too narrow for `LOGO_LINES`."""

LOGO_WIDTH = max(len(line) for line in LOGO_LINES)
