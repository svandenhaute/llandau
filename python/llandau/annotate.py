"""Brackets with a label, under or over (a part of) a formula or anything else.

    e = eq("cal(S) = {length} - {legendre}", length="cal(L)_g [gamma]", legendre="epsilon.alt T")
    e["length"].set_color(BLUE)
    bracket(e["length"], "geodesic\nlength")                   # blue too: the target's colour
    bracket(e["legendre"], "Legendre\npair", at=e["length"])   # at the same height as the first
    bracket(e["length"], eq("2 epsilon.alt", size=SMALL))      # a label can be any mobject
"""
from __future__ import annotations

import numpy as np

from .mobject import DOWN, Mobject, VGroup, VMobject
from .shapes import Line, Polyline
from . import theme
from .text import _MATH, _plain, typst

__all__ = ["bracket"]


def bracket(target: Mobject, label: str | Mobject | None = None, direction=DOWN, at: Mobject | float | None = None,
            gap: float = 0.08, tick: float = 0.1, label_gap: float = 0.08, size: float | None = None, color=None,
            stroke_width: float = 2) -> VGroup:
    """A square bracket spanning target's width, under it (direction=DOWN) or over it (UP), and
    label centred beyond it: VGroup(bracket, label).

    label        text as text() takes it, its lines (split at "\\n") centred, or a mobject (eq(...), an image, ...)
    at           a mobject or a y: the bracket keeps gap from its edge instead of target's, e.g.
                 another part of the formula, so that several brackets sit at one height
    gap          from the target to the bracket; tick: the length of its ends, 0: an underline
    label_gap    from the bracket to the label; size: of a text label, default the theme's small
    color        default the target's (its first path's), else the theme's
    stroke_width in manim's units (hundredths of a frame unit)
    """
    d = np.asarray(direction, dtype=float)
    if d[0] != 0 or abs(d[1]) != 1:
        raise ValueError("bracket: direction is DOWN or UP")
    side = d[1]
    ref = target if at is None else at
    edge = ref.get_critical_point(d)[1] if isinstance(ref, Mobject) else float(ref)
    y = edge + side * gap
    x0, x1 = target.get_left()[0], target.get_right()[0]
    if color is None:
        color = _color(target)
    if tick:
        shape = Polyline([x0, y, 0], [x0, y + side * tick, 0], [x1, y + side * tick, 0], [x1, y, 0],
                         color=color, stroke_width=stroke_width)
    else:
        shape = Line([x0, y, 0], [x1, y, 0], color=color, stroke_width=stroke_width)
    if label is None:
        return VGroup(shape)
    if isinstance(label, str):  # as text() would, its lines centred and close
        lines = label.replace("\n", " \\\n") if _MATH.search(label) else _plain(label)
        label = typst("#set align(center)\n#set par(leading: 0.35em)\n" + lines,
                      size=size if size is not None else theme.current().small, color=color)
    label.next_to(shape, d, buff=label_gap)
    return VGroup(shape, label)


def _color(target: Mobject):
    for m in target.family_members_with_points():
        if isinstance(m, VMobject):
            return m.get_color()
    return theme.current().fg
