"""Colours, font and type scale. Change these in one place for the whole deck."""
from dataclasses import dataclass

from .mobject import BLACK, WHITE, ManimColor

__all__ = [
    "Theme", "DARK", "LIGHT", "use", "current",
    "FONT", "TITLE", "BODY", "SMALL", "LABEL",
]

FONT = "Linux Biolinum O"

# effective font sizes (what you see on screen, no x10 hack needed)
TITLE = 34
BODY = 28
SMALL = 22
LABEL = 20


@dataclass(frozen=True)
class Theme:
    bg: ManimColor
    fg: ManimColor


DARK = Theme(bg=BLACK, fg=WHITE)
LIGHT = Theme(bg=WHITE, fg=BLACK)

_current = DARK


def use(theme: Theme) -> None:
    """Set the theme that text() & co. take their default colour from (Deck does this in setup)."""
    global _current
    _current = theme


def current() -> Theme:
    return _current
