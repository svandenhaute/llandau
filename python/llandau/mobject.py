"""Mobjects without manim: the subset of manim's model that slides use.

Positions are in frame units (the frame is 14.22 x 8, origin at its center,
y up), points are 3D like manim's so code written for manim keeps working.
A VMobject holds cubic Bézier curves (4 points per curve) and a fill and
stroke style; placement methods (next_to, move_to, arrange, ...) follow
manim's definitions, including measuring bounds over all control points.
"""
from __future__ import annotations

import copy as _copy
import math
import re
from pathlib import Path

import numpy as np

__all__ = [
    "config", "ORIGIN", "UP", "DOWN", "LEFT", "RIGHT", "UL", "UR", "DL", "DR", "IN", "OUT",
    "PI", "TAU", "DEGREES", "SMALL_BUFF", "MED_SMALL_BUFF", "MED_LARGE_BUFF", "LARGE_BUFF",
    "DEFAULT_STROKE_WIDTH", "Color", "ManimColor", "interpolate_color",
    "WHITE", "BLACK", "GRAY", "GREY", "LIGHT_GRAY", "LIGHT_GREY", "DARK_GRAY", "DARK_GREY",
    "RED", "GREEN", "BLUE", "YELLOW", "ORANGE", "PURPLE", "PINK", "TEAL", "GOLD", "MAROON",
    "Mobject", "VMobject", "Group", "VGroup", "ImageMobject", "SVGMobject", "VectorizedPoint",
]


class _Config:
    """The frame (manim's defaults)."""
    frame_height = 8.0
    frame_width = 8.0 * 16 / 9
    pixel_width = 1920
    pixel_height = 1080

    @property
    def frame_x_radius(self):
        return self.frame_width / 2

    @property
    def frame_y_radius(self):
        return self.frame_height / 2


config = _Config()

ORIGIN = np.array([0.0, 0.0, 0.0])
UP = np.array([0.0, 1.0, 0.0])
DOWN = -UP
RIGHT = np.array([1.0, 0.0, 0.0])
LEFT = -RIGHT
OUT = np.array([0.0, 0.0, 1.0])
IN = -OUT
UL, UR, DL, DR = UP + LEFT, UP + RIGHT, DOWN + LEFT, DOWN + RIGHT
PI = np.pi
TAU = 2 * np.pi
DEGREES = TAU / 360
SMALL_BUFF, MED_SMALL_BUFF, MED_LARGE_BUFF, LARGE_BUFF = 0.1, 0.25, 0.5, 1.0
DEFAULT_STROKE_WIDTH = 4.0


# --- colors -----------------------------------------------------------------------

class Color:
    """An sRGB color: Color("#0074d9"), Color((1, 0, 0)), Color(other)."""

    def __init__(self, value=None, alpha: float = 1.0):
        if value is None:
            value = (0.0, 0.0, 0.0)
        if isinstance(value, Color):
            self.rgb, self.alpha = value.rgb.copy(), value.alpha
            return
        if isinstance(value, str):
            name = value.strip()
            if name in _MATPLOTLIB:
                name = _MATPLOTLIB[name]
            elif name.upper() in _NAMED:
                name = _NAMED[name.upper()]
            h = name.lstrip("#")
            if len(h) == 3:
                h = "".join(c * 2 for c in h)
            if len(h) not in (6, 8) or not re.fullmatch(r"[0-9a-fA-F]+", h):
                rgba = _matplotlib_color(value)
                if rgba is None:
                    raise ValueError(f"not a color: {value!r}")
                self.rgb, self.alpha = np.array(rgba[:3]), rgba[3] if len(rgba) > 3 else alpha
                return
            self.rgb = np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])
            self.alpha = int(h[6:8], 16) / 255 if len(h) == 8 else alpha
            return
        v = np.asarray(value, dtype=float).ravel()
        if v.max(initial=0) > 1:
            v = v / 255
        self.rgb = v[:3].copy()
        self.alpha = float(v[3]) if len(v) > 3 else alpha

    def to_rgb(self) -> np.ndarray:
        return self.rgb.copy()

    def to_rgba(self) -> np.ndarray:
        return np.append(self.rgb, self.alpha)

    def to_hex(self) -> str:
        return "#" + "".join(f"{round(255 * c):02X}" for c in self.rgb)

    def interpolate(self, other, alpha: float) -> "Color":
        other = Color(other)
        return Color(np.append(self.rgb + alpha * (other.rgb - self.rgb), self.alpha + alpha * (other.alpha - self.alpha)))

    def __eq__(self, other):
        try:
            other = Color(other)
        except (ValueError, TypeError):
            return NotImplemented
        return np.allclose(self.rgb, other.rgb) and math.isclose(self.alpha, other.alpha)

    def __hash__(self):
        return hash(self.to_hex())

    def __repr__(self):
        return f"Color({self.to_hex()!r})"

    def __str__(self):
        return self.to_hex()


ManimColor = Color

# manim's palette
_NAMED = {
    "WHITE": "#FFFFFF", "BLACK": "#000000", "GRAY": "#888888", "GREY": "#888888",
    "LIGHT_GRAY": "#BBBBBB", "LIGHT_GREY": "#BBBBBB", "DARK_GRAY": "#444444", "DARK_GREY": "#444444",
    "RED": "#FC6255", "GREEN": "#83C167", "BLUE": "#58C4DD", "YELLOW": "#FFFF00",
    "ORANGE": "#FF862F", "PURPLE": "#9A72AC", "PINK": "#D147BD", "TEAL": "#5CD0B3",
    "GOLD": "#F0AC5F", "MAROON": "#C55F73",
}
# matplotlib's one-letter colors, its cycle (C0...) and common CSS names
_MATPLOTLIB = {
    "k": "#000000", "w": "#FFFFFF", "r": "#FF0000", "g": "#008000", "b": "#0000FF", "c": "#00BFBF",
    "m": "#BF00BF", "y": "#BFBF00",
    **{f"C{i}": c for i, c in enumerate(["#0074D9", "#E76852", "#63BB8D", "#A03825", "#8E6CC2", "#F2A541",
                                          "#3CB4C8", "#6F6F6F"])},
    "black": "#000000", "white": "#FFFFFF", "gray": "#808080", "grey": "#808080", "lightgray": "#D3D3D3",
    "lightgrey": "#D3D3D3", "darkgray": "#A9A9A9", "darkgrey": "#A9A9A9", "red": "#FF0000", "green": "#008000",
    "blue": "#0000FF", "orange": "#FFA500", "purple": "#800080", "none": "#00000000",
}


def _matplotlib_color(value):
    """Any other matplotlib color, when matplotlib is installed."""
    try:
        from matplotlib.colors import to_rgba
        return to_rgba(value)
    except (ImportError, ValueError):
        return None


WHITE, BLACK, GRAY, GREY = Color("WHITE"), Color("BLACK"), Color("GRAY"), Color("GREY")
LIGHT_GRAY = LIGHT_GREY = Color("LIGHT_GRAY")
DARK_GRAY = DARK_GREY = Color("DARK_GRAY")
RED, GREEN, BLUE, YELLOW = Color("RED"), Color("GREEN"), Color("BLUE"), Color("YELLOW")
ORANGE, PURPLE, PINK, TEAL = Color("ORANGE"), Color("PURPLE"), Color("PINK"), Color("TEAL")
GOLD, MAROON = Color("GOLD"), Color("MAROON")


def interpolate_color(a, b, alpha: float) -> Color:
    return Color(a).interpolate(b, alpha)


# --- mobjects ---------------------------------------------------------------------

def _rotation(angle: float, axis) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    return np.array([
        [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
        [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
        [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
    ])


class Mobject:
    """Points and submobjects; everything that can be placed and animated."""

    def __init__(self, z_index: float = 0, **kwargs):
        self.submobjects: list[Mobject] = []
        self.points = np.zeros((0, 3))
        self.z_index = z_index

    # --- structure --------------------------------------------------------------

    def add(self, *mobjects):
        for m in mobjects:
            if m is self:
                raise ValueError("a mobject cannot contain itself")
            if not isinstance(m, Mobject):
                raise TypeError(f"only mobjects can be added, not {type(m).__name__}")
        for m in mobjects:
            if m in self.submobjects:
                self.submobjects.remove(m)
            self.submobjects.append(m)
        return self

    def add_to_back(self, *mobjects):
        self.remove(*mobjects)
        self.submobjects[0:0] = list(mobjects)
        return self

    def remove(self, *mobjects):
        for m in mobjects:
            if m in self.submobjects:
                self.submobjects.remove(m)
        return self

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.get_group_class()(*self.submobjects[index])
        return self.submobjects[index]

    def __iter__(self):
        return iter(self.submobjects)

    def __len__(self):
        return len(self.submobjects)

    def get_group_class(self):
        return Group

    def get_family(self) -> list[Mobject]:
        family = [self]
        for m in self.submobjects:
            family.extend(m.get_family())
        return family

    def family_members_with_points(self) -> list[Mobject]:
        return [m for m in self.get_family() if m.has_points()]

    def has_points(self) -> bool:
        return len(self.points) > 0

    def has_no_points(self) -> bool:
        return not self.has_points()

    def copy(self):
        return _copy.deepcopy(self)

    def __deepcopy__(self, memo):
        result = self.__class__.__new__(self.__class__)
        memo[id(self)] = result
        shared = getattr(self, "_shared_attributes", ())
        for key, value in self.__dict__.items():
            if key in ("target", "_saved_state"):
                continue
            setattr(result, key, value if key in shared else _copy.deepcopy(value, memo))
        return result

    def become(self, other: Mobject):
        """Take the points, style and structure of `other` (a copy of it)."""
        other = other.copy()
        for key, value in other.__dict__.items():
            setattr(self, key, value)
        return self

    def set_z_index(self, z_index: float, family: bool = True):
        for m in self.get_family() if family else [self]:
            m.z_index = z_index
        return self

    def generate_target(self):
        self.target = self.copy()
        return self.target

    def save_state(self):
        self._saved_state = self.copy()
        return self

    def restore(self):
        if getattr(self, "_saved_state", None) is None:
            raise ValueError("restore() without save_state()")
        saved = self._saved_state
        self.become(saved)
        self._saved_state = saved
        return self

    @property
    def animate(self):
        from .animation import _AnimationBuilder
        return _AnimationBuilder(self)

    # --- points ------------------------------------------------------------------

    def get_all_points(self) -> np.ndarray:
        points = [m.points for m in self.get_family() if m.has_points()]
        return np.concatenate(points) if points else np.zeros((0, 3))

    def apply_points_function(self, function, about_point=None, about_edge=None):
        if about_point is None and about_edge is not None:
            about_point = self.get_critical_point(about_edge)
        for m in self.family_members_with_points():
            if about_point is None:
                m.points = function(m.points)
            else:
                m.points = function(m.points - about_point) + about_point
        return self

    def shift(self, *vectors):
        total = np.sum([np.asarray(v, dtype=float) for v in vectors], axis=0)
        for m in self.family_members_with_points():
            m.points = m.points + total
        return self

    def scale(self, factor: float, about_point=None, about_edge=ORIGIN, scale_stroke: bool = False, **kwargs):
        if about_point is None:
            about_point = self.get_critical_point(about_edge)
        # strokes of SVG paths keep their width relative to the drawing
        for m in self.get_family():
            if scale_stroke or getattr(m, "stroke_scales", False):
                if hasattr(m, "stroke_width"):
                    m.stroke_width *= abs(factor)
        return self.apply_points_function(lambda p: p * factor, about_point=about_point)

    def stretch(self, factor: float, dim: int, about_point=None, about_edge=ORIGIN, scale_stroke: bool = False):
        if about_point is None:
            about_point = self.get_critical_point(about_edge)
        # strokes of SVG paths scale with the drawing: by the geometric mean of the
        # factors along x and y (as SVG transforms scale them)
        for m in self.get_family():
            if scale_stroke or getattr(m, "stroke_scales", False):
                if hasattr(m, "stroke_width"):
                    m.stroke_width *= math.sqrt(abs(factor))

        def function(points):
            points = points.copy()
            points[:, dim] *= factor
            return points
        return self.apply_points_function(function, about_point=about_point)

    def rotate(self, angle: float, axis=OUT, about_point=None, about_edge=ORIGIN):
        if about_point is None:
            about_point = self.get_critical_point(about_edge)
        matrix = _rotation(angle, axis)
        return self.apply_points_function(lambda p: p @ matrix.T, about_point=about_point)

    def flip(self, axis=UP, **kwargs):
        return self.rotate(PI, axis, **kwargs)

    def apply_matrix(self, matrix, about_point=ORIGIN):
        matrix = np.asarray(matrix, dtype=float)
        full = np.identity(3)
        full[: matrix.shape[0], : matrix.shape[1]] = matrix
        return self.apply_points_function(lambda p: p @ full.T, about_point=about_point)

    def apply_function(self, function, about_point=ORIGIN):
        return self.apply_points_function(lambda p: np.apply_along_axis(function, 1, p), about_point=about_point)

    # --- bounds --------------------------------------------------------------------

    def get_critical_point(self, direction) -> np.ndarray:
        """The point of the bounding box in `direction` (ORIGIN: center, UL: corner, ...)."""
        direction = np.asarray(direction, dtype=float)
        points = self.get_all_points()
        result = np.zeros(3)
        if len(points) == 0:
            return result
        lo, hi = points.min(axis=0), points.max(axis=0)
        for dim in range(3):
            d = direction[dim] if dim < len(direction) else 0
            result[dim] = hi[dim] if d > 0 else lo[dim] if d < 0 else (lo[dim] + hi[dim]) / 2
        return result

    def get_edge_center(self, direction):
        return self.get_critical_point(direction)

    def get_corner(self, direction):
        return self.get_critical_point(direction)

    def get_center(self):
        return self.get_critical_point(ORIGIN)

    def get_center_of_mass(self):
        return self.get_all_points().mean(axis=0)

    def get_top(self):
        return self.get_critical_point(UP)

    def get_bottom(self):
        return self.get_critical_point(DOWN)

    def get_left(self):
        return self.get_critical_point(LEFT)

    def get_right(self):
        return self.get_critical_point(RIGHT)

    def get_zenith(self):
        return self.get_critical_point(OUT)

    def get_x(self):
        return self.get_center()[0]

    def get_y(self):
        return self.get_center()[1]

    def get_z(self):
        return self.get_center()[2]

    def length_over_dim(self, dim: int) -> float:
        points = self.get_all_points()
        return float(np.ptp(points[:, dim])) if len(points) else 0.0

    @property
    def width(self) -> float:
        return self.length_over_dim(0)

    @width.setter
    def width(self, value):
        self.scale_to_fit_width(value)

    @property
    def height(self) -> float:
        return self.length_over_dim(1)

    @height.setter
    def height(self, value):
        self.scale_to_fit_height(value)

    def get_width(self):
        return self.width

    def get_height(self):
        return self.height

    def rescale_to_fit(self, length: float, dim: int, stretch: bool = False, **kwargs):
        old = self.length_over_dim(dim)
        if old == 0:
            return self
        if stretch:
            return self.stretch(length / old, dim, **kwargs)
        return self.scale(length / old, **kwargs)

    def scale_to_fit_width(self, width, **kwargs):
        return self.rescale_to_fit(width, 0, **kwargs)

    def scale_to_fit_height(self, height, **kwargs):
        return self.rescale_to_fit(height, 1, **kwargs)

    def stretch_to_fit_width(self, width, **kwargs):
        return self.rescale_to_fit(width, 0, stretch=True, **kwargs)

    def stretch_to_fit_height(self, height, **kwargs):
        return self.rescale_to_fit(height, 1, stretch=True, **kwargs)

    def match_width(self, other, **kwargs):
        return self.scale_to_fit_width(other.width, **kwargs)

    def match_height(self, other, **kwargs):
        return self.scale_to_fit_height(other.height, **kwargs)

    def set(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self

    # --- placement ------------------------------------------------------------------

    def center(self):
        return self.shift(-self.get_center())

    def move_to(self, point_or_mobject, aligned_edge=ORIGIN, coor_mask=np.array([1, 1, 1])):
        if isinstance(point_or_mobject, Mobject):
            target = point_or_mobject.get_critical_point(aligned_edge)
        else:
            target = np.asarray(point_or_mobject, dtype=float)
        return self.shift((target - self.get_critical_point(aligned_edge)) * coor_mask)

    def next_to(self, mobject_or_point, direction=RIGHT, buff=MED_SMALL_BUFF, aligned_edge=ORIGIN,
                submobject_to_align=None, index_of_submobject_to_align=None, coor_mask=np.array([1, 1, 1])):
        direction = np.asarray(direction, dtype=float)
        aligned_edge = np.asarray(aligned_edge, dtype=float)
        if isinstance(mobject_or_point, Mobject):
            target_aligner = mobject_or_point
            if index_of_submobject_to_align is not None:
                target_aligner = mobject_or_point[index_of_submobject_to_align]
            target = target_aligner.get_critical_point(aligned_edge + direction)
        else:
            target = np.asarray(mobject_or_point, dtype=float)
        aligner = self
        if submobject_to_align is not None:
            aligner = submobject_to_align
        elif index_of_submobject_to_align is not None:
            aligner = self[index_of_submobject_to_align]
        point = aligner.get_critical_point(aligned_edge - direction)
        return self.shift((target - point + buff * direction) * coor_mask)

    def align_on_border(self, direction, buff=MED_LARGE_BUFF):
        direction = np.asarray(direction, dtype=float)
        target = np.sign(direction) * np.array([config.frame_x_radius, config.frame_y_radius, 0])
        shift = target - self.get_critical_point(direction) - buff * direction
        return self.shift(shift * np.abs(np.sign(direction)))

    def to_corner(self, corner=DL, buff=MED_LARGE_BUFF):
        return self.align_on_border(corner, buff)

    def to_edge(self, edge=LEFT, buff=MED_LARGE_BUFF):
        return self.align_on_border(edge, buff)

    def align_to(self, mobject_or_point, direction=ORIGIN):
        if isinstance(mobject_or_point, Mobject):
            point = mobject_or_point.get_critical_point(direction)
        else:
            point = np.asarray(mobject_or_point, dtype=float)
        direction = np.asarray(direction, dtype=float)
        for dim in range(3):
            if direction[dim] != 0:
                self.set_coord(point[dim], dim, direction)
        return self

    def set_coord(self, value, dim: int, direction=ORIGIN):
        current = self.get_critical_point(direction)[dim]
        shift = np.zeros(3)
        shift[dim] = value - current
        return self.shift(shift)

    def set_x(self, x, direction=ORIGIN):
        return self.set_coord(x, 0, direction)

    def set_y(self, y, direction=ORIGIN):
        return self.set_coord(y, 1, direction)

    def set_z(self, z, direction=ORIGIN):
        return self.set_coord(z, 2, direction)

    def surround(self, mobject, dim_to_match: int = 0, stretch: bool = False, buff=MED_SMALL_BUFF):
        self.replace(mobject, dim_to_match, stretch)
        length = mobject.length_over_dim(dim_to_match)
        if length:
            self.scale((length + buff) / length)
        return self

    def replace(self, mobject, dim_to_match: int = 0, stretch: bool = False):
        if stretch:
            self.stretch_to_fit_width(mobject.width)
            self.stretch_to_fit_height(mobject.height)
        else:
            self.rescale_to_fit(mobject.length_over_dim(dim_to_match), dim_to_match)
        return self.shift(mobject.get_center() - self.get_center())

    def arrange(self, direction=RIGHT, buff=MED_SMALL_BUFF, center: bool = True, **kwargs):
        for m1, m2 in zip(self.submobjects, self.submobjects[1:]):
            m2.next_to(m1, direction, buff, **kwargs)
        if center:
            self.center()
        return self

    def arrange_in_grid(self, rows=None, cols=None, buff=MED_SMALL_BUFF, cell_alignment=ORIGIN,
                        row_alignments=None, col_alignments=None, row_heights=None, col_widths=None,
                        flow_order: str = "rd", **kwargs):
        mobs = self.submobjects.copy()
        start = self.get_center()

        def init_size(num, alignments, sizes):
            if num is not None:
                return num
            if alignments is not None:
                return len(alignments)
            if sizes is not None:
                return len(sizes)
            return None
        cols = init_size(cols, col_alignments, col_widths)
        rows = init_size(rows, row_alignments, row_heights)
        if rows is None and cols is None:
            cols = math.ceil(math.sqrt(len(mobs)))
        if rows is None:
            rows = math.ceil(len(mobs) / cols)
        if cols is None:
            cols = math.ceil(len(mobs) / rows)
        if rows * cols < len(mobs):
            raise ValueError("too few rows and columns for all submobjects")
        buff_x, buff_y = buff if isinstance(buff, tuple) else (buff, buff)
        cell_alignment = np.asarray(cell_alignment, dtype=float)

        def init_alignments(alignments, num, mapping, dir_):
            if alignments is None:
                return [cell_alignment * dir_] * num
            return [mapping[a] for a in alignments]
        row_al = init_alignments(row_alignments, rows, {"u": UP, "c": ORIGIN, "d": DOWN}, RIGHT)
        col_al = init_alignments(col_alignments, cols, {"l": LEFT, "c": ORIGIN, "r": RIGHT}, UP)
        mapper = {
            "dr": lambda r, c: (rows - r - 1) + c * rows,
            "dl": lambda r, c: (rows - r - 1) + (cols - c - 1) * rows,
            "ur": lambda r, c: r + c * rows,
            "ul": lambda r, c: r + (cols - c - 1) * rows,
            "rd": lambda r, c: (rows - r - 1) * cols + c,
            "ld": lambda r, c: (rows - r - 1) * cols + (cols - c - 1),
            "ru": lambda r, c: r * cols + c,
            "lu": lambda r, c: r * cols + (cols - c - 1),
        }
        order = mapper[flow_order]
        row_al = row_al[::-1]
        row_heights = list(row_heights)[::-1] if row_heights is not None else None
        placeholder = Mobject()
        mobs.extend([placeholder] * (rows * cols - len(mobs)))
        grid = [[mobs[order(r, c)] for c in range(cols)] for r in range(rows)]
        heights = [max(grid[r][c].height for c in range(cols)) for r in range(rows)]
        widths = [max(grid[r][c].width for r in range(rows)) for c in range(cols)]
        if row_heights is not None:
            heights = [h if h is not None else m for h, m in zip(row_heights, heights)]
        if col_widths is not None:
            widths = [w if w is not None else m for w, m in zip(col_widths, widths)]
        y = 0.0
        for r in range(rows):
            x = 0.0
            for c in range(cols):
                if grid[r][c] is not placeholder:
                    alignment = row_al[r] + col_al[c]
                    cell = VMobject().set_points(np.array([[x, y, 0], [x + widths[c], y + heights[r], 0]]))
                    grid[r][c].move_to(cell, alignment)
                x += widths[c] + buff_x
            y += heights[r] + buff_y
        return self.move_to(start)

    # --- style (overridden by VMobject and ImageMobject) -----------------------------

    def set_color(self, color, family: bool = True):
        for m in self.submobjects if family else []:
            m.set_color(color)
        return self

    def set_opacity(self, opacity: float, family: bool = True):
        for m in self.submobjects if family else []:
            m.set_opacity(opacity)
        return self

    def fade(self, darkness: float = 0.5, family: bool = True):
        for m in self.submobjects if family else []:
            m.fade(darkness)
        return self

    def get_color(self):
        for m in self.get_family()[1:]:
            return m.get_color()
        return WHITE

    def match_style(self, other, family: bool = True):
        return self

    def interpolate(self, start, end, alpha: float):
        """Become `alpha` of the way from `start` to `end` (same structure)."""
        self.points = start.points + alpha * (end.points - start.points)
        return self

    def __repr__(self):
        return f"{type(self).__name__}({len(self.submobjects)} submobjects)"


class Group(Mobject):
    def __init__(self, *mobjects, **kwargs):
        super().__init__(**kwargs)
        self.add(*mobjects)


class VMobject(Mobject):
    """Cubic Bézier curves with a fill and a stroke. stroke_width is in manim's
    units (hundredths of a frame unit)."""

    def __init__(self, color=None, fill_color=None, fill_opacity: float = 0.0, stroke_color=None,
                 stroke_opacity: float = 1.0, stroke_width: float = DEFAULT_STROKE_WIDTH, z_index: float = 0,
                 joint_type: str = "miter", cap_style: str = "butt", stacked: bool = False, **kwargs):
        super().__init__(z_index=z_index)
        self.joint_type = joint_type  # miter, round or bevel
        self.cap_style = cap_style  # butt, round or square
        self.stacked = stacked  # each subpath painted (fill, then stroke) over the ones before
        color = Color(color) if color is not None else WHITE
        self.fill_color = Color(fill_color) if fill_color is not None else color
        self.fill_opacity = fill_opacity
        self.stroke_color = Color(stroke_color) if stroke_color is not None else color
        self.stroke_opacity = stroke_opacity
        self.stroke_width = stroke_width
        self.generate_points()

    def generate_points(self):
        pass

    def get_group_class(self):
        return VGroup

    # --- style -------------------------------------------------------------------

    def set_fill(self, color=None, opacity: float | None = None, family: bool = True):
        for m in self.get_family() if family else [self]:
            if isinstance(m, VMobject):
                if color is not None:
                    m.fill_color = Color(color)
                if opacity is not None:
                    m.fill_opacity = opacity
        return self

    def set_stroke(self, color=None, width: float | None = None, opacity: float | None = None, family: bool = True,
                   **kwargs):
        for m in self.get_family() if family else [self]:
            if isinstance(m, VMobject):
                if color is not None:
                    m.stroke_color = Color(color)
                if width is not None:
                    m.stroke_width = width
                if opacity is not None:
                    m.stroke_opacity = opacity
        return self

    def set_style(self, fill_color=None, fill_opacity=None, stroke_color=None, stroke_width=None,
                  stroke_opacity=None, family: bool = True, **kwargs):
        self.set_fill(fill_color, fill_opacity, family)
        self.set_stroke(stroke_color, stroke_width, stroke_opacity, family)
        return self

    def set_color(self, color, family: bool = True):
        self.set_fill(color, family=family)
        self.set_stroke(color, family=family)
        for m in self.get_family() if family else []:
            if not isinstance(m, VMobject):
                m.set_color(color)
        return self

    def set_opacity(self, opacity: float, family: bool = True):
        self.set_fill(opacity=opacity, family=family)
        self.set_stroke(opacity=opacity, family=family)
        return self

    def fade(self, darkness: float = 0.5, family: bool = True):
        for m in self.get_family() if family else [self]:
            if isinstance(m, VMobject):
                m.fill_opacity *= 1 - darkness
                m.stroke_opacity *= 1 - darkness
            elif m is not self:
                m.fade(darkness, family=False)
        return self

    def get_fill_color(self):
        return self.fill_color

    def get_fill_opacity(self):
        return self.fill_opacity

    def get_stroke_color(self):
        return self.stroke_color

    def get_stroke_width(self):
        return self.stroke_width

    def get_stroke_opacity(self):
        return self.stroke_opacity

    def get_color(self):
        if self.stroke_width == 0 or self.stroke_opacity == 0:
            return self.fill_color
        return self.stroke_color

    color = property(get_color, lambda self, c: self.set_color(c))

    def match_style(self, other, family: bool = True):
        if isinstance(other, VMobject):
            self.set_style(other.fill_color, other.fill_opacity, other.stroke_color, other.stroke_width,
                           other.stroke_opacity, family=False)
        if family:
            for m1, m2 in zip(self.submobjects, other.submobjects):
                m1.match_style(m2)
        return self

    def style_array(self) -> np.ndarray:
        """fill RGBA, stroke RGBA, stroke width in frame units, join, cap, stacked (the renderer's style)."""
        return np.concatenate([
            self.fill_color.rgb, [self.fill_color.alpha * self.fill_opacity],
            self.stroke_color.rgb, [self.stroke_color.alpha * self.stroke_opacity],
            [0.01 * self.stroke_width,
             _JOINS.get(getattr(self, "joint_type", "miter"), 0), _CAPS.get(getattr(self, "cap_style", "butt"), 0),
             float(getattr(self, "stacked", False))],
        ]).astype(np.float32)

    # --- points ------------------------------------------------------------------

    def set_points(self, points):
        self.points = np.array(points, dtype=float).reshape(-1, 3)
        return self

    def append_points(self, points):
        self.points = np.concatenate([self.points, np.asarray(points, dtype=float).reshape(-1, 3)])
        return self

    def clear_points(self):
        self.points = np.zeros((0, 3))
        return self

    def start_new_path(self, point):
        # a lone point: the next curve starts there
        self.append_points([point])
        return self

    def _pending_start(self):
        """The start of the next curve: a lone point, or the last point."""
        return self.points[-1] if len(self.points) else None

    def _complete(self, curve):
        if len(self.points) % 4 == 1:
            self.points = self.points[:-1]
        self.append_points(curve)

    def add_cubic_bezier_curve_to(self, handle1, handle2, anchor):
        start = self._pending_start()
        if start is None:
            raise ValueError("start_new_path first")
        self._complete([start, handle1, handle2, anchor])
        return self

    def add_quadratic_bezier_curve_to(self, handle, anchor):
        start = self._pending_start()
        handle, anchor = np.asarray(handle, dtype=float), np.asarray(anchor, dtype=float)
        return self.add_cubic_bezier_curve_to(start + 2 / 3 * (handle - start), anchor + 2 / 3 * (handle - anchor), anchor)

    def add_line_to(self, point):
        start = self._pending_start()
        point = np.asarray(point, dtype=float)
        self._complete([start + t * (point - start) for t in (0, 1 / 3, 2 / 3, 1)])
        return self

    def add_points_as_corners(self, points):
        for p in points:
            self.add_line_to(p)
        return self

    def close_path(self):
        if not self.is_closed():
            subpath = self.get_subpaths()[-1]
            self.add_line_to(subpath[0])
        return self

    def set_points_as_corners(self, points):
        points = np.asarray(points, dtype=float)
        if len(points) < 2:
            return self.set_points(points)
        curves = [[a + t * (b - a) for t in (0, 1 / 3, 2 / 3, 1)] for a, b in zip(points[:-1], points[1:])]
        return self.set_points(np.array(curves).reshape(-1, 3))

    def set_points_smoothly(self, points):
        self.set_points_as_corners(points)
        return self.make_smooth()

    def make_smooth(self):
        new = []
        for subpath in self.get_subpaths():
            anchors = np.concatenate([subpath[0::4], subpath[-1:]])
            h1, h2 = smooth_handles(anchors)
            for i in range(len(anchors) - 1):
                new.extend([anchors[i], h1[i], h2[i], anchors[i + 1]])
        return self.set_points(np.array(new).reshape(-1, 3))

    def get_num_curves(self) -> int:
        return len(self.points) // 4

    def get_subpaths(self) -> list[np.ndarray]:
        """Runs of curves where each starts at the previous one's end."""
        points = self.points[: 4 * self.get_num_curves()]
        if len(points) == 0:
            return []
        starts = points[0::4]
        ends = points[3::4]
        breaks = [0] + [i for i in range(1, len(starts)) if not np.allclose(starts[i], ends[i - 1], atol=1e-6)]
        breaks.append(len(starts))
        return [points[4 * a: 4 * b] for a, b in zip(breaks[:-1], breaks[1:])]

    def is_closed(self) -> bool:
        return len(self.points) >= 4 and np.allclose(self.points[0], self.points[4 * self.get_num_curves() - 1])

    def get_start(self):
        return self.points[0].copy()

    def get_end(self):
        return self.points[-1].copy()

    def get_anchors(self):
        if len(self.points) < 4:
            return self.points.copy()
        return np.concatenate([self.points[0::4], self.points[3::4]]).reshape(2, -1, 3).transpose(1, 0, 2).reshape(-1, 3)

    def get_start_anchors(self):
        return self.points[0::4]

    def get_end_anchors(self):
        return self.points[3::4]

    def _curve_lengths(self, samples: int = 10) -> np.ndarray:
        t = np.linspace(0, 1, samples + 1)[:, None]
        lengths = []
        for c in self.points[: 4 * self.get_num_curves()].reshape(-1, 4, 3):
            p = bezier_points(c, t)
            lengths.append(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
        return np.array(lengths)

    def get_arc_length(self) -> float:
        return float(self._curve_lengths().sum())

    def point_from_proportion(self, alpha: float) -> np.ndarray:
        """The point `alpha` of the way along the curves, by length."""
        lengths = self._curve_lengths()
        if len(lengths) == 0:
            raise ValueError("no curves")
        target = alpha * lengths.sum()
        current = 0.0
        curves = self.points[: 4 * len(lengths)].reshape(-1, 4, 3)
        for curve, length in zip(curves, lengths):
            if current + length >= target:
                residue = (target - current) / length if length else 0.0
                return bezier_points(curve, np.array([[residue]]))[0]
            current += length
        return curves[-1][-1].copy()

    def reverse_points(self):
        self.points = self.points[::-1].copy()
        return self

    def pointwise_become_partial(self, vmobject: VMobject, a: float, b: float):
        n = vmobject.get_num_curves()
        if n == 0 or b <= a:
            return self.clear_points()
        if a <= 0 and b >= 1:
            return self.set_points(vmobject.points)
        curves = vmobject.points[: 4 * n].reshape(-1, 4, 3)
        i0, r0 = min(int(a * n), n - 1), a * n - min(int(a * n), n - 1)
        i1, r1 = min(int(b * n), n - 1), b * n - min(int(b * n), n - 1)
        if i0 == i1:
            new = [sub_bezier(curves[i0], r0, r1)]
        else:
            new = [sub_bezier(curves[i0], r0, 1)] + list(curves[i0 + 1: i1]) + [sub_bezier(curves[i1], 0, r1)]
        return self.set_points(np.concatenate(new))

    def get_points_defining_boundary(self):
        return self.get_all_points()

    def interpolate(self, start, end, alpha: float):
        super().interpolate(start, end, alpha)
        if isinstance(start, VMobject) and isinstance(end, VMobject):
            self.fill_color = start.fill_color.interpolate(end.fill_color, alpha)
            self.stroke_color = start.stroke_color.interpolate(end.stroke_color, alpha)
            mix = lambda a, b: a + alpha * (b - a)  # noqa: E731
            self.fill_opacity = mix(start.fill_opacity, end.fill_opacity)
            self.stroke_opacity = mix(start.stroke_opacity, end.stroke_opacity)
            self.stroke_width = mix(start.stroke_width, end.stroke_width)
        return self


_JOINS = {"miter": 0, "round": 1, "bevel": 2}
_CAPS = {"butt": 0, "round": 1, "square": 2, "projecting": 2}


class VGroup(VMobject):
    def __init__(self, *vmobjects, **kwargs):
        super().__init__(**kwargs)
        self.add(*vmobjects)

    def add(self, *vmobjects):
        flat = []
        for m in vmobjects:
            if isinstance(m, (list, tuple)):
                flat.extend(m)
            else:
                flat.append(m)
        return super().add(*flat)


class VectorizedPoint(VMobject):
    def __init__(self, location=ORIGIN, **kwargs):
        super().__init__(stroke_width=0, **kwargs)
        self.set_points([location])


# --- Bézier helpers ----------------------------------------------------------------

def bezier_points(curve, t) -> np.ndarray:
    """Points of a cubic at parameters t ((k, 1))."""
    p0, p1, p2, p3 = curve
    return ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * (t ** 2) * p2 + (t ** 3) * p3


def split_bezier(curve, t):
    p0, p1, p2, p3 = np.asarray(curve, dtype=float)
    a, b, c = p0 + t * (p1 - p0), p1 + t * (p2 - p1), p2 + t * (p3 - p2)
    d, e = a + t * (b - a), b + t * (c - b)
    f = d + t * (e - d)
    return np.array([p0, a, d, f]), np.array([f, e, c, p3])


def sub_bezier(curve, t0: float, t1: float) -> np.ndarray:
    if t1 <= 0:
        return np.repeat(np.asarray(curve, dtype=float)[:1], 4, axis=0)
    first = split_bezier(curve, t1)[0]
    return split_bezier(first, t0 / t1)[1]


def smooth_handles(anchors) -> tuple[np.ndarray, np.ndarray]:
    """Handles making the curves through `anchors` twice differentiable (a
    natural cubic spline; periodic when the anchors close), as manim does."""
    a = np.asarray(anchors, dtype=float)
    n = len(a) - 1
    if n < 1:
        return np.zeros((0, 3)), np.zeros((0, 3))
    if n == 1:
        return a[:1] + (a[1:] - a[:1]) / 3, a[:1] + 2 * (a[1:] - a[:1]) / 3
    # manim's is_closed: its tolerance is negative for negative coordinates,
    # which then count as open; kept, so that shapes come out the same
    tolerance = 1e-8 + 1e-5 * a[0]
    if np.all(np.abs(a[-1] - a[0]) <= tolerance):
        return _closed_handles(a)
    m = np.zeros((n, n))
    rhs = np.zeros((n, a.shape[1]))
    m[0, 0], m[0, 1] = 2, 1
    rhs[0] = a[0] + 2 * a[1]
    for i in range(1, n - 1):
        m[i, i - 1], m[i, i], m[i, i + 1] = 1, 4, 1
        rhs[i] = 4 * a[i] + 2 * a[i + 1]
    m[n - 1, n - 2], m[n - 1, n - 1] = 2, 7
    rhs[n - 1] = 8 * a[n - 1] + a[n]
    h1 = np.linalg.solve(m, rhs)
    h2 = np.empty_like(h1)
    h2[:-1] = 2 * a[1:n] - h1[1:]
    h2[-1] = (a[n] + h1[-1]) / 2
    return h1, h2


def _closed_handles(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """manim's get_smooth_closed_cubic_bezier_handle_points, as it is."""
    n = len(a) - 1
    cp, up = np.empty(max(n - 1, 1)), np.empty(max(n - 1, 1))
    cp[0], up[0] = 1 / 3, 1 / 3
    for i in range(1, n - 1):
        cp[i] = 1 / (4 - cp[i - 1])
        up[i] = -cp[i] * up[i - 1]
    cp_last = 1 / (3 - cp[n - 2])
    up_last = cp_last * (1 - up[n - 2])
    q = np.empty((n, a.shape[1]))
    q[n - 1] = up_last
    for i in range(n - 2, -1, -1):
        q[i] = up[i] - cp[i] * q[i + 1]
    aux = 4 * a[:n] + 2 * a[1:]
    y = np.empty((n, a.shape[1]))
    y[0] = aux[0] / 3
    for i in range(1, n - 1):
        y[i] = cp[i] * (aux[i] - y[i - 1])
    y[n - 1] = cp_last * (aux[n - 1] - y[n - 2])
    for i in range(n - 2, -1, -1):
        y[i] = y[i] - cp[i] * y[i + 1]
    h1 = y - 1 / (1 + q[0] + q[n - 1]) * q * (y[0] + y[n - 1])
    h2 = np.empty((n, a.shape[1]))
    h2[: n - 1] = 2 * a[1:n] - h1[1:n]
    h2[n - 1] = 2 * a[n] - h1[0]
    return h1, h2


# --- images and SVGs ------------------------------------------------------------------

class ImageMobject(Mobject):
    """A raster image (PNG or JPEG file, or an RGBA array (h, w, 4) uint8).
    Its height is pixels / 1080 × 8 frame units by default, as in manim."""

    def __init__(self, filename_or_array, scale_to_resolution: int = 1080, opacity: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        if isinstance(filename_or_array, (str, Path)):
            from ._core import image_size
            self.path = str(filename_or_array)
            self.pixels = None
            width, height = image_size(self.path)
        else:
            self.path = None
            self.pixels = np.ascontiguousarray(np.asarray(filename_or_array, dtype=np.uint8))
            if self.pixels.ndim == 2:
                self.pixels = np.repeat(self.pixels[..., None], 3, axis=2)
            if self.pixels.shape[2] == 3:
                self.pixels = np.concatenate([self.pixels, np.full(self.pixels.shape[:2] + (1,), 255, np.uint8)], 2)
            height, width = self.pixels.shape[:2]
        self.pixel_size = (width, height)
        h = height / scale_to_resolution * config.frame_height
        w = h * width / height
        self.points = np.array([[-w / 2, h / 2, 0], [w / 2, h / 2, 0], [-w / 2, -h / 2, 0], [w / 2, -h / 2, 0]])
        self.opacity = opacity

    def __deepcopy__(self, memo):
        # the pixels are shared, not copied
        pixels, self.pixels = self.pixels, None
        try:
            result = super().__deepcopy__(memo)
        finally:
            self.pixels = pixels
        result.pixels = pixels
        return result

    def set_opacity(self, opacity: float, family: bool = True):
        self.opacity = opacity
        return super().set_opacity(opacity, family)

    def fade(self, darkness: float = 0.5, family: bool = True):
        self.opacity *= 1 - darkness
        return super().fade(darkness, family)

    def get_color(self):
        return WHITE

    def interpolate(self, start, end, alpha: float):
        super().interpolate(start, end, alpha)
        self.opacity = start.opacity + alpha * (end.opacity - start.opacity)
        return self


def _view_box_scale(data: bytes, size) -> float:
    """SVG user units per unit of usvg's size (pixels), so that coordinates come out
    in the file's own units, as manim (svgelements) reads them."""
    head = data[:4096].decode("utf-8", "replace")
    m = re.search(r"<svg\b[^>]*\bviewBox\s*=\s*[\"']([^\"']+)[\"']", head, re.S)
    if not m:
        return 1.0
    values = [float(v) for v in re.split(r"[\s,]+", m.group(1).strip())]
    return values[2] / size[0] if size[0] else 1.0


class SVGMobject(VMobject):
    """An SVG file as one VMobject per drawn path, flipped to y up and centered;
    scaled to `height` (or `width`; both None: the file's own units).
    id_to_vgroup_dict maps group ids to the paths inside them."""

    def __init__(self, file_name, height: float | None = 2, width: float | None = None, should_center: bool = True,
                 stroke_width: float | None = None, fill_opacity: float | None = None, color=None, **kwargs):
        super().__init__(stroke_width=0)
        from ._core import parse_svg
        self.file_name = str(file_name)
        data = Path(self.file_name).read_bytes()
        svg = parse_svg(data)
        scale = _view_box_scale(data, svg["size"])
        self.svg_size = np.array(svg["size"]) * scale
        self.id_to_vgroup_dict: dict[str, VGroup] = {}
        for path in svg["paths"]:
            points = np.zeros((len(path["points"]), 3))
            points[:, :2] = path["points"] * scale
            points[:, 1] *= -1
            fill, stroke = path["fill"], path["stroke"]
            # without a stroke: manim's default stroke (white, opaque, zero width)
            m = VMobject(
                fill_color=Color(fill[:3]), fill_opacity=fill[3],
                stroke_color=Color(stroke[:3]) if stroke[3] > 0 else WHITE, stroke_opacity=stroke[3] if stroke[3] > 0 else 1.0,
                stroke_width=100 * path["stroke_width"] * scale if stroke[3] > 0 else 0.0,
            ).set_points(points)
            m.stroke_scales = stroke_width is None
            if stroke_width is not None:
                m.stroke_width = stroke_width
            if fill_opacity is not None:
                m.fill_opacity = fill_opacity
            if color is not None:
                m.set_color(color)
            self.add(m)
            for group_id in path["groups"]:
                self.id_to_vgroup_dict.setdefault(group_id, VGroup()).add(m)
        if should_center:
            self.center()
        if height is not None:
            self.scale_to_fit_height(height)
        if width is not None:
            self.scale_to_fit_width(width)
