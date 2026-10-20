"""Geometric shapes, built the way manim builds them."""
from __future__ import annotations

import numpy as np

from .mobject import (
    BLACK, DEFAULT_STROKE_WIDTH, DL, DR, LEFT, MED_SMALL_BUFF, ORIGIN, OUT, PI, RED, RIGHT, SMALL_BUFF, TAU,
    UL, UP, UR, WHITE, YELLOW, Mobject, VGroup, VMobject,
)

__all__ = [
    "Arc", "Circle", "Dot", "Ellipse", "Annulus", "Sector", "Line", "DashedLine", "Arrow", "Vector", "DoubleArrow",
    "Polygon", "Polyline", "RegularPolygon", "Triangle", "Rectangle", "Square", "RoundedRectangle",
    "SurroundingRectangle", "BackgroundRectangle", "Cross", "DashedVMobject", "Underline",
]


def _unit(angle: float) -> np.ndarray:
    return np.array([np.cos(angle), np.sin(angle), 0.0])


def _arc_points(radius: float, start_angle: float, angle: float, num_components: int, center) -> np.ndarray:
    """manim's Arc: anchors on the circle, handles along the tangents."""
    angles = np.linspace(start_angle, start_angle + angle, num_components)
    anchors = np.array([radius * _unit(a) for a in angles]) + np.asarray(center, dtype=float)
    tangents = np.array([radius * _unit(a + PI / 2) for a in angles])
    d_theta = angle / (num_components - 1)
    h1 = anchors[:-1] + (d_theta / 3) * tangents[:-1]
    h2 = anchors[1:] - (d_theta / 3) * tangents[1:]
    points = np.empty((4 * (num_components - 1), 3))
    points[0::4], points[1::4], points[2::4], points[3::4] = anchors[:-1], h1, h2, anchors[1:]
    return points


class Arc(VMobject):
    def __init__(self, radius: float = 1.0, start_angle: float = 0.0, angle: float = TAU / 4,
                 num_components: int = 9, arc_center=ORIGIN, **kwargs):
        self.radius, self.start_angle, self.angle = radius, start_angle, angle
        self.num_components, self.arc_center = num_components, np.asarray(arc_center, dtype=float)
        super().__init__(**kwargs)

    def generate_points(self):
        self.set_points(_arc_points(self.radius, self.start_angle, self.angle, self.num_components, self.arc_center))


class Circle(Arc):
    def __init__(self, radius: float = 1.0, color=RED, **kwargs):
        super().__init__(radius=radius, start_angle=0, angle=TAU, color=color, **kwargs)

    def point_at_angle(self, angle: float):
        return self.get_center() + self.width / 2 * _unit(angle)

    def surround(self, mobject, dim_to_match: int = 0, stretch: bool = False, buffer_factor: float = 1.2):
        self.replace(mobject, dim_to_match, stretch)
        self.width = np.sqrt(mobject.width ** 2 + mobject.height ** 2)
        return self.scale(buffer_factor)


class Dot(Circle):
    def __init__(self, point=ORIGIN, radius: float = 0.08, stroke_width: float = 0, fill_opacity: float = 1.0,
                 color=WHITE, **kwargs):
        super().__init__(radius=radius, stroke_width=stroke_width, fill_opacity=fill_opacity, color=color, **kwargs)
        self.move_to(point)


class Ellipse(Circle):
    def __init__(self, width: float = 2, height: float = 1, **kwargs):
        super().__init__(**kwargs)
        self.stretch_to_fit_width(width)
        self.stretch_to_fit_height(height)


class Annulus(VMobject):
    def __init__(self, inner_radius: float = 1, outer_radius: float = 2, fill_opacity: float = 1, stroke_width: float = 0,
                 color=WHITE, **kwargs):
        self.inner_radius, self.outer_radius = inner_radius, outer_radius
        super().__init__(fill_opacity=fill_opacity, stroke_width=stroke_width, color=color, **kwargs)

    def generate_points(self):
        outer = _arc_points(self.outer_radius, 0, TAU, 9, ORIGIN)
        inner = _arc_points(self.inner_radius, 0, -TAU, 9, ORIGIN)
        self.set_points(np.concatenate([outer, inner]))


class Sector(VMobject):
    def __init__(self, radius: float = 1, angle: float = TAU / 4, start_angle: float = 0, fill_opacity: float = 1,
                 stroke_width: float = 0, color=WHITE, arc_center=ORIGIN, **kwargs):
        self.radius, self.angle, self.start_angle, self.arc_center = radius, angle, start_angle, arc_center
        super().__init__(fill_opacity=fill_opacity, stroke_width=stroke_width, color=color, **kwargs)

    def generate_points(self):
        arc = _arc_points(self.radius, self.start_angle, self.angle, 9, self.arc_center)
        self.set_points(arc)
        self.add_line_to(np.asarray(self.arc_center, dtype=float))
        self.add_line_to(arc[0])


def _pointify(point_or_mobject, direction=None):
    if isinstance(point_or_mobject, Mobject):
        if direction is None:
            return point_or_mobject.get_center()
        return point_or_mobject.get_critical_point(direction)
    return np.asarray(point_or_mobject, dtype=float)


class Line(VMobject):
    def __init__(self, start=LEFT, end=RIGHT, buff: float = 0, path_arc: float = 0, **kwargs):
        self.buff, self.path_arc = buff, path_arc
        self._start, self._end = start, end
        super().__init__(**kwargs)

    def generate_points(self):
        start, end = _pointify(self._start), _pointify(self._end)
        if self.path_arc:
            # an arc through both points
            chord = end - start
            length = np.linalg.norm(chord)
            radius = length / (2 * np.sin(self.path_arc / 2))
            mid = (start + end) / 2
            normal = np.cross(OUT, chord / length)
            center = mid + normal * radius * np.cos(self.path_arc / 2)
            start_angle = np.arctan2(*(start - center)[1::-1])
            self.set_points(_arc_points(radius, start_angle, self.path_arc, 9, center))
        else:
            self.set_points_as_corners([start, end])
        self._apply_buff()

    def _apply_buff(self):
        if self.buff == 0:
            return
        length = self.get_length()
        if length < 2 * self.buff:
            return
        ratio = self.buff / length
        self.pointwise_become_partial(self.copy(), ratio, 1 - ratio)

    def put_start_and_end_on(self, start, end):
        current_start, current_end = self.get_start(), self.get_end()
        if np.allclose(current_start, current_end):
            self._start, self._end = start, end
            self.generate_points()
            return self
        start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
        old, new = current_end - current_start, end - start
        scale = np.linalg.norm(new) / np.linalg.norm(old)
        angle = np.arctan2(new[1], new[0]) - np.arctan2(old[1], old[0])
        self.scale(scale, about_point=current_start)
        self.rotate(angle, about_point=current_start)
        return self.shift(start - current_start)

    def get_vector(self):
        return self.get_end() - self.get_start()

    def get_unit_vector(self):
        v = self.get_vector()
        n = np.linalg.norm(v)
        return v / n if n else v

    def get_angle(self) -> float:
        v = self.get_vector()
        return float(np.arctan2(v[1], v[0]))

    def get_length(self) -> float:
        return float(np.linalg.norm(self.get_vector()))

    def set_length(self, length: float):
        return self.scale(length / self.get_length())


class DashedLine(Line):
    def __init__(self, *args, dash_length: float = 0.05, dashed_ratio: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        num_dashes = max(2, int(np.ceil(self.get_length() / dash_length * dashed_ratio)))
        dashes = DashedVMobject(self, num_dashes=num_dashes, dashed_ratio=dashed_ratio)
        self.clear_points()
        self.add(*dashes)


class _Tip(VMobject):
    """A filled triangle pointing along +x from its base, tip at the origin."""

    def __init__(self, length: float, width: float, color=WHITE, **kwargs):
        super().__init__(fill_opacity=1, stroke_width=0, color=color, **kwargs)
        self.set_points_as_corners([ORIGIN, [-length, width / 2, 0], [-length, -width / 2, 0], ORIGIN])


class Arrow(Line):
    """A line with a filled triangular tip, shortened by `buff` at both ends.
    Thinner and with a smaller tip when short, as in manim."""

    def __init__(self, start=LEFT, end=RIGHT, buff: float = MED_SMALL_BUFF, stroke_width: float = 6,
                 tip_length: float = 0.35, max_tip_length_to_length_ratio: float = 0.25,
                 max_stroke_width_to_length_ratio: float = 5, tip_shape=None, **kwargs):
        self.tip_length = tip_length
        self.max_tip_length_to_length_ratio = max_tip_length_to_length_ratio
        self.max_stroke_width_to_length_ratio = max_stroke_width_to_length_ratio
        super().__init__(start, end, buff=buff, stroke_width=stroke_width, **kwargs)
        self._add_tip(self.get_end())

    def _add_tip(self, at, reverse: bool = False):
        length = self.get_length()
        tip_length = min(self.tip_length, self.max_tip_length_to_length_ratio * length)
        self.stroke_width = min(self.stroke_width, self.max_stroke_width_to_length_ratio * length)
        direction = -self.get_unit_vector() if reverse else self.get_unit_vector()
        tip = _Tip(tip_length, tip_length, color=self.stroke_color)
        tip.rotate(np.arctan2(direction[1], direction[0]), about_point=ORIGIN).shift(at)
        # the line stops at the tip's base
        if reverse:
            start = self.get_start() + direction * tip_length
            self.set_points_as_corners([start, self.get_end()])
        else:
            self.set_points_as_corners([self.get_start(), self.get_end() - direction * tip_length])
        self.add(tip)
        return tip

    def get_end(self):
        tips = [m for m in self.submobjects if isinstance(m, _Tip)]
        return tips[-1].points[0].copy() if tips else super().get_end()


class Vector(Arrow):
    def __init__(self, direction=RIGHT, buff: float = 0, **kwargs):
        direction = np.asarray(direction, dtype=float)
        if len(direction) == 2:
            direction = np.append(direction, 0)
        super().__init__(ORIGIN, direction, buff=buff, **kwargs)


class DoubleArrow(Arrow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._add_tip(self.points[0], reverse=True)


class Polyline(VMobject):
    def __init__(self, *vertices, **kwargs):
        self.vertices = [np.asarray(v, dtype=float) for v in vertices]
        super().__init__(**kwargs)

    def generate_points(self):
        self.set_points_as_corners(self.vertices)


class Polygon(VMobject):
    def __init__(self, *vertices, **kwargs):
        self.vertices = [np.asarray(v, dtype=float) for v in vertices]
        super().__init__(**kwargs)

    def generate_points(self):
        self.set_points_as_corners([*self.vertices, self.vertices[0]])

    def get_vertices(self):
        return self.get_start_anchors()

    def round_corners(self, radius: float = 0.5):
        vertices = self.get_start_anchors()
        n = len(vertices)
        self.clear_points()
        for i in range(n):
            prev, v, nxt = vertices[i - 1], vertices[i], vertices[(i + 1) % n]
            a = (prev - v) / np.linalg.norm(prev - v)
            b = (nxt - v) / np.linalg.norm(nxt - v)
            angle = np.arccos(np.clip(np.dot(a, b), -1, 1))
            cut = radius / np.tan(angle / 2)
            p, q = v + a * cut, v + b * cut
            # a circular arc from p to q: handles at the standard cubic approximation
            turn = PI - angle
            k = 4 / 3 * np.tan(turn / 4) * radius
            if len(self.points) == 0:
                self.start_new_path(p)
            else:
                self.add_line_to(p)
            self.add_cubic_bezier_curve_to(p - a * k, q - b * k, q)
        self.add_line_to(self.points[0])
        return self


class RegularPolygon(Polygon):
    def __init__(self, n: int = 6, radius: float = 1, start_angle: float | None = None, **kwargs):
        if start_angle is None:
            start_angle = (n % 2) * PI / 2
        vertices = [radius * _unit(start_angle + k * TAU / n) for k in range(n)]
        super().__init__(*vertices, **kwargs)


class Triangle(RegularPolygon):
    def __init__(self, **kwargs):
        super().__init__(n=3, **kwargs)


class Rectangle(Polygon):
    def __init__(self, width: float = 4.0, height: float = 2.0, color=WHITE, **kwargs):
        super().__init__(UR, UL, DL, DR, color=color, **kwargs)
        self.stretch_to_fit_width(width)
        self.stretch_to_fit_height(height)


class Square(Rectangle):
    def __init__(self, side_length: float = 2.0, **kwargs):
        super().__init__(width=side_length, height=side_length, **kwargs)


class RoundedRectangle(Rectangle):
    def __init__(self, corner_radius: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.round_corners(min(corner_radius, self.width / 2, self.height / 2))


class SurroundingRectangle(RoundedRectangle):
    def __init__(self, mobject, color=YELLOW, buff: float = SMALL_BUFF, corner_radius: float = 0.0, **kwargs):
        width, height = mobject.width + 2 * buff, mobject.height + 2 * buff
        if corner_radius:
            super().__init__(corner_radius=corner_radius, width=width, height=height, color=color, **kwargs)
        else:
            Rectangle.__init__(self, width=width, height=height, color=color, **kwargs)
        self.move_to(mobject)


class BackgroundRectangle(SurroundingRectangle):
    def __init__(self, mobject, color=BLACK, stroke_width: float = 0, fill_opacity: float = 0.75, buff: float = 0,
                 **kwargs):
        super().__init__(mobject, color=color, buff=buff, stroke_width=stroke_width, fill_opacity=fill_opacity,
                         **kwargs)


class Cross(VGroup):
    def __init__(self, mobject=None, stroke_color=RED, stroke_width: float = 6, scale_factor: float = 1, **kwargs):
        super().__init__(Line(UL, DR), Line(UR, DL), **kwargs)
        if mobject is not None:
            self.replace(mobject, stretch=True)
        self.scale(scale_factor)
        self.set_stroke(color=stroke_color, width=stroke_width)


class Underline(Line):
    def __init__(self, mobject, buff: float = SMALL_BUFF, **kwargs):
        super().__init__(LEFT, RIGHT, **kwargs)
        self.match_width(mobject)
        self.next_to(mobject, np.array([0, -1, 0]), buff=buff)


class DashedVMobject(VMobject):
    """Dashes along a VMobject's curves (by curve count, as manim)."""

    def __init__(self, vmobject: VMobject, num_dashes: int = 15, dashed_ratio: float = 0.5, color=None, **kwargs):
        super().__init__(color=color or vmobject.stroke_color, stroke_width=vmobject.stroke_width, **kwargs)
        r = dashed_ratio
        period = 1 / num_dashes
        for i in range(num_dashes):
            a = i * period
            dash = VMobject().match_style(vmobject).pointwise_become_partial(vmobject, a, a + r * period)
            self.add(dash)
        if color is not None:
            self.set_color(color)
