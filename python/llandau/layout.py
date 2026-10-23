"""Named screen regions and relative placement.

    place(img, R.body.right)                  # fit into the right half of the content area
    place(label, R.body.ul, anchor=UL)        # pin to the upper-left corner of the upper-left quadrant
    place(fig, R.body.right_of(img, gap=M))   # whatever space is left to the right of img
    below(caption, img, gap=S)                # left-aligned under img
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from .mobject import config, Mobject, VMobject, Group, VGroup, ORIGIN, LEFT, RIGHT, UP, DOWN, RED
from .shapes import Rectangle, DashedVMobject

__all__ = [
    "Region", "R", "place", "region_of",
    "below", "above", "left_of", "right_of", "align", "align_x", "align_y",
    "stack", "row", "group",
    "XS", "S", "M", "L", "XL",
]

# spacing scale, in frame units (frame is 14.2 x 8)
XS, S, M, L, XL = 0.1, 0.25, 0.5, 1.0, 2.0


@dataclass(frozen=True)
class Region:
    """Axis-aligned rectangle in frame coordinates that knows how to split itself."""
    x0: float
    x1: float
    y0: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def center(self) -> np.ndarray:
        return np.array([(self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2, 0.0])

    def point(self, direction=ORIGIN) -> np.ndarray:
        """Point on the region: ORIGIN = center, LEFT = middle of left edge, UL = corner, ..."""
        d = np.asarray(direction, dtype=float)
        return self.center + np.array([d[0] * self.width / 2, d[1] * self.height / 2, 0.0])

    # --- splitting ---------------------------------------------------------

    def hsplit(self, *weights: float, gap: float = 0) -> list[Region]:
        """Split into columns, left to right. hsplit(0.4) -> 40% | 60%; hsplit(1, 2, 1) -> weights."""
        bounds = _split(self.x0, self.x1, weights, gap)
        return [Region(a, b, self.y0, self.y1) for a, b in bounds]

    def vsplit(self, *weights: float, gap: float = 0) -> list[Region]:
        """Split into rows, top to bottom. vsplit(0.3) -> top 30% / bottom 70%."""
        bounds = _split(self.y1, self.y0, weights, gap)
        return [Region(self.x0, self.x1, b, a) for a, b in bounds]

    def cols(self, n: int, gap: float = 0) -> list[Region]:
        return self.hsplit(*[1] * n, gap=gap)

    def rows(self, n: int, gap: float = 0) -> list[Region]:
        return self.vsplit(*[1] * n, gap=gap)

    def grid(self, nrows: int, ncols: int, gap: float = 0) -> list[Region]:
        """Cells in reading order (row by row)."""
        return [cell for r in self.rows(nrows, gap) for cell in r.cols(ncols, gap)]

    @property
    def left(self) -> Region:
        return self.hsplit(0.5)[0]

    @property
    def right(self) -> Region:
        return self.hsplit(0.5)[1]

    @property
    def top(self) -> Region:
        return self.vsplit(0.5)[0]

    @property
    def bottom(self) -> Region:
        return self.vsplit(0.5)[1]

    @property
    def ul(self) -> Region:
        return self.top.left

    @property
    def ur(self) -> Region:
        return self.top.right

    @property
    def dl(self) -> Region:
        return self.bottom.left

    @property
    def dr(self) -> Region:
        return self.bottom.right

    def inset(self, dx: float, dy: float | None = None) -> Region:
        dy = dx if dy is None else dy
        return Region(self.x0 + dx, self.x1 - dx, self.y0 + dy, self.y1 - dy)

    # --- the space left over next to something ------------------------------

    def left_of(self, other, gap: float = 0) -> Region:
        return Region(self.x0, _bounds(other)[0] - gap, self.y0, self.y1)

    def right_of(self, other, gap: float = 0) -> Region:
        return Region(_bounds(other)[1] + gap, self.x1, self.y0, self.y1)

    def above(self, other, gap: float = 0) -> Region:
        return Region(self.x0, self.x1, _bounds(other)[3] + gap, self.y1)

    def below(self, other, gap: float = 0) -> Region:
        return Region(self.x0, self.x1, self.y0, _bounds(other)[2] - gap)

    # --- convenience -------------------------------------------------------

    def place(self, mob, **kwargs):
        return place(mob, self, **kwargs)

    def outline(self, color=RED) -> VMobject:
        """Dashed rectangle showing the region, for debugging a layout."""
        rect = Rectangle(width=self.width, height=self.height, stroke_color=color, stroke_width=2)
        return DashedVMobject(rect.move_to(self.center), num_dashes=60)


def _split(start: float, stop: float, weights, gap: float) -> list[tuple[float, float]]:
    if len(weights) == 1 and 0 < weights[0] < 1:
        weights = (weights[0], 1 - weights[0])
    total = stop - start
    sign = 1 if total >= 0 else -1
    usable = abs(total) - gap * (len(weights) - 1)
    out, pos = [], start
    for w in weights:
        size = usable * w / sum(weights)
        out.append((pos, pos + sign * size))
        pos += sign * (size + gap)
    return out


def _bounds(obj) -> tuple[float, float, float, float]:
    """(x0, x1, y0, y1) of a Region or Mobject."""
    if isinstance(obj, Region):
        return obj.x0, obj.x1, obj.y0, obj.y1
    return obj.get_left()[0], obj.get_right()[0], obj.get_bottom()[1], obj.get_top()[1]


def region_of(mob: Mobject, buff: float = 0) -> Region:
    """Region covering a mobject, e.g. region_of(img).ur to put a label in its upper-right quadrant."""
    x0, x1, y0, y1 = _bounds(mob)
    return Region(x0 - buff, x1 + buff, y0 - buff, y1 + buff)


class _Frame:
    """Standard regions of the 16:9 frame.

    full   whole frame
    safe   frame minus a margin
    title  band at the top of `safe`, where title() puts slide titles
    body   `safe` minus the title band (plus a gap) and a footer strip (slide number)
    footer the strip below `body` (citations, the slide number at its right end)
    """
    margin = 0.3
    title_height = 0.8
    title_gap = 0.4  # whitespace between title band and body
    footer_height = 0.35

    @property
    def full(self) -> Region:
        w, h = config.frame_width, config.frame_height
        return Region(-w / 2, w / 2, -h / 2, h / 2)

    @property
    def safe(self) -> Region:
        return self.full.inset(self.margin)

    @property
    def title(self) -> Region:
        s = self.safe
        return Region(s.x0, s.x1, s.y1 - self.title_height, s.y1)

    @property
    def body(self) -> Region:
        s = self.safe
        return Region(s.x0, s.x1, s.y0 + self.footer_height, s.y1 - self.title_height - self.title_gap)

    @property
    def footer(self) -> Region:
        s = self.safe
        return Region(s.x0, s.x1, s.y0, s.y0 + self.footer_height)

    def __getattr__(self, name):  # R.left, R.ul, R.hsplit(...) etc. act on the full frame
        return getattr(self.full, name)


R = _Frame()


def place(mob, region: Region, anchor=ORIGIN, fit=True, buff: float = 0, scale: float | None = None):
    """Position (and optionally resize) mob inside region.

    anchor  which point of the region (and of mob) to align: ORIGIN, LEFT, UL, ...
    fit     True: shrink to fit if too large; "fill": scale up or down to fit exactly; False: keep size
    buff    padding inside the region
    scale   explicit scale factor (overrides fit)
    """
    region = region.inset(buff) if buff else region
    if scale is not None:
        mob.scale(scale)
    elif fit:
        factor = min(region.width / mob.width, region.height / mob.height)
        if fit == "fill" or factor < 1:
            mob.scale(factor)
    anchor = np.asarray(anchor, dtype=float)
    mob.move_to(region.point(anchor), aligned_edge=anchor)
    return mob


# --- relative placement ------------------------------------------------------

def below(mob, ref, gap: float = S, align=LEFT):
    return mob.next_to(ref, DOWN, buff=gap, aligned_edge=align)


def above(mob, ref, gap: float = S, align=LEFT):
    return mob.next_to(ref, UP, buff=gap, aligned_edge=align)


def left_of(mob, ref, gap: float = S, align=ORIGIN):
    return mob.next_to(ref, LEFT, buff=gap, aligned_edge=align)


def right_of(mob, ref, gap: float = S, align=ORIGIN):
    return mob.next_to(ref, RIGHT, buff=gap, aligned_edge=align)


def align(mob, ref, edge, by=None):
    """Line up one edge of mob with the same edge of ref, moving along that axis only.
    align(a, b, LEFT) -> same left edge; align(a, b, UP) -> same top edge.
    by: a part of mob whose edge lines up instead, e.g. the left-hand sides of two formulas so
    their = signs line up: align(e2, e1["lhs"], RIGHT, by=e2["lhs"]); ref can be a part too."""
    edge = np.asarray(edge, dtype=float)
    anchor = mob if by is None else by
    delta = (ref.get_critical_point(edge) - anchor.get_critical_point(edge)) * np.abs(edge)
    return mob.shift(delta)


def align_x(mob, ref):
    return mob.set_x(ref.get_x())


def align_y(mob, ref):
    return mob.set_y(ref.get_y())


def group(*mobs):
    """VGroup when possible (so styling methods work), Group otherwise (e.g. with images)."""
    if len(mobs) == 1 and isinstance(mobs[0], (list, tuple)):
        mobs = tuple(mobs[0])
    if all(isinstance(m, VMobject) for m in mobs):
        return VGroup(*mobs)
    return Group(*mobs)


def stack(*mobs, gap: float = M, align=LEFT):
    """Vertical list, top to bottom."""
    return group(*mobs).arrange(DOWN, buff=gap, aligned_edge=align)


def row(*mobs, gap: float = M, align=ORIGIN):
    """Horizontal list, left to right."""
    return group(*mobs).arrange(RIGHT, buff=gap, aligned_edge=align)
