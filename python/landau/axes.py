"""manim's coordinate systems: NumberLine, Axes, NumberPlane, and graphs
(ParametricFunction, FunctionGraph), with manim's defaults and placement.
Numbers and axis labels are typeset with Typst (math). Colors default to the
deck's foreground.

For matplotlib-style plots of data, see landau.plot.
"""
from __future__ import annotations

import numpy as np

from . import theme
from .mobject import (
    DOWN, LEFT, MED_SMALL_BUFF, ORIGIN, PI, RIGHT, SMALL_BUFF, UP, UR, YELLOW, BLUE, GREEN, Color, Mobject, VGroup,
    VMobject, config,
)
from .shapes import DashedLine, Dot, Line, Polygon, _Tip

__all__ = ["NumberLine", "Axes", "NumberPlane", "ParametricFunction", "FunctionGraph"]

DEFAULT_ARROW_TIP_LENGTH = 0.35


def _decimal_places(step: float) -> int:
    step = abs(float(step))
    if step == 0 or float(step).is_integer():
        return 0
    return max(0, int(np.ceil(-np.log10(step - np.floor(step) if step % 1 else step) - 1e-9)))


def _number_text(value: float, places: int) -> str:
    s = f"{value:.{places}f}"
    if s.startswith("-"):
        s = "−" + s[1:]
    return s


def _labels(strings, size):
    from .text import texts
    return texts([f"${s}$" for s in strings], size=size)


class NumberLine(Line):
    """A line from x_range[0] to x_range[1] with ticks every x_range[2]; centered
    at the origin, as in manim."""

    def __init__(self, x_range=None, length: float | None = None, unit_size: float = 1, include_ticks: bool = True,
                 tick_size: float = 0.1, numbers_with_elongated_ticks=None, longer_tick_multiple: int = 2,
                 exclude_origin_tick: bool = False, rotation: float = 0, stroke_width: float = 2.0,
                 include_tip: bool = False, tip_width: float = DEFAULT_ARROW_TIP_LENGTH,
                 tip_height: float = DEFAULT_ARROW_TIP_LENGTH, include_numbers: bool = False, font_size: float = 36,
                 label_direction=DOWN, line_to_number_buff: float = MED_SMALL_BUFF, decimal_number_config=None,
                 numbers_to_exclude=None, numbers_to_include=None, color=None, **kwargs):
        if x_range is None:
            x_range = [round(-config.frame_x_radius), round(config.frame_x_radius), 1]
        elif len(x_range) == 2:
            x_range = [*x_range, 1]
        self.x_range = np.array(x_range, dtype=float)
        self.x_min, self.x_max, self.x_step = self.x_range
        self.tick_size = tick_size
        self.numbers_with_elongated_ticks = list(numbers_with_elongated_ticks or [])
        self.longer_tick_multiple = longer_tick_multiple
        self.exclude_origin_tick = exclude_origin_tick
        self.font_size = font_size
        self.label_direction = np.asarray(label_direction, dtype=float)
        self.line_to_number_buff = line_to_number_buff
        places = (decimal_number_config or {}).get("num_decimal_places", _decimal_places(self.x_step))
        self.decimal_places = places
        self.numbers_to_exclude = list(numbers_to_exclude or [])
        color = color if color is not None else theme.current().fg
        super().__init__(self.x_min * RIGHT, self.x_max * RIGHT, stroke_width=stroke_width, color=color, **kwargs)
        if length:
            self.scale(length / (self.x_max - self.x_min))
        else:
            self.scale(unit_size)
        self.center()
        # where x_min and x_max are: an invisible curve between them that moves with the
        # line (a tip shortens the drawn line, not the scale)
        self._marker = VMobject(stroke_width=0, fill_opacity=0).set_points(
            [self.get_start(), self.get_start(), self.get_end(), self.get_end()])
        self.add(self._marker)
        self.ticks = VGroup()
        self.tip = None
        if include_tip:
            start, end = self._ends
            self.tip = _Tip(tip_height, tip_width, color=color)
            self.tip.shift(end)
            self.set_points_as_corners([start, end - tip_height * RIGHT])
            self.add(self.tip)
        if include_ticks:
            self.add_ticks()
        self.rotate(rotation, about_point=ORIGIN)
        self.numbers = VGroup()
        if include_numbers or numbers_to_include is not None:
            self.add_numbers(numbers_to_include, excluding=self.numbers_to_exclude)

    @property
    def _ends(self):
        return self._marker.points[0], self._marker.points[3]

    def get_tick_range(self) -> np.ndarray:
        ticks = np.arange(self.x_min, self.x_max + self.x_step / 1e3, self.x_step)
        if self.exclude_origin_tick:
            ticks = ticks[np.abs(ticks) > 1e-9]
        return ticks

    def get_unit_vector(self):
        a, b = self._ends
        return (b - a) / (self.x_max - self.x_min)

    def get_unit_size(self) -> float:
        return float(np.linalg.norm(self.get_unit_vector()))

    def number_to_point(self, number):
        number = np.asarray(number, dtype=float)
        a, b = self._ends
        alpha = (number - self.x_min) / (self.x_max - self.x_min)
        return a + np.multiply.outer(alpha, b - a) if number.ndim else a + alpha * (b - a)

    n2p = number_to_point

    def point_to_number(self, point):
        a, b = self._ends
        v = b - a
        alpha = np.dot(np.asarray(point, dtype=float) - a, v) / np.dot(v, v)
        return self.x_min + alpha * (self.x_max - self.x_min)

    p2n = point_to_number

    def get_tick(self, x: float, size: float | None = None) -> Line:
        size = self.tick_size if size is None else size
        direction = self.get_unit_vector()
        normal = np.array([-direction[1], direction[0], 0.0])
        normal /= np.linalg.norm(normal)
        center = self.number_to_point(x)
        return Line(center - size * normal, center + size * normal, stroke_width=self.stroke_width,
                    color=self.stroke_color)

    def add_ticks(self):
        ticks = VGroup()
        for x in self.get_tick_range():
            size = self.tick_size
            if np.any(np.isclose(x, self.numbers_with_elongated_ticks)):
                size *= self.longer_tick_multiple
            ticks.add(self.get_tick(x, size))
        self.add(ticks)
        self.ticks = ticks
        return self

    def add_numbers(self, x_values=None, excluding=None, font_size: float | None = None, **kwargs):
        if x_values is None:
            x_values = self.get_tick_range()
        excluding = list(self.numbers_to_exclude if excluding is None else excluding)
        values = [x for x in x_values if not np.any(np.isclose(x, excluding))]
        labels = _labels([_number_text(x, self.decimal_places) for x in values], font_size or self.font_size)
        numbers = VGroup()
        for x, label in zip(values, labels):
            label.next_to(self.number_to_point(x), self.label_direction, buff=self.line_to_number_buff)
            numbers.add(label)
        self.add(numbers)
        self.numbers = numbers
        return numbers

    def add_labels(self, dict_values: dict, direction=None, buff: float | None = None, font_size=None):
        direction = self.label_direction if direction is None else direction
        buff = self.line_to_number_buff if buff is None else buff
        labels = VGroup()
        for x, label in dict_values.items():
            if isinstance(label, str):
                label = _labels([label], font_size or self.font_size)[0]
            labels.add(label.next_to(self.number_to_point(x), direction, buff=buff))
        self.add(labels)
        self.labels = labels
        return labels


def _origin_shift(lo: float, hi: float) -> float:
    return lo if lo > 0 else hi if hi < 0 else 0.0


class Axes(VGroup):
    """manim's Axes: two number lines crossing at the origin (or the nearest
    end), centered on the screen."""

    def __init__(self, x_range=None, y_range=None, x_length: float | None = round(config.frame_width) - 2,
                 y_length: float | None = round(config.frame_height) - 2, axis_config=None, x_axis_config=None,
                 y_axis_config=None, tips: bool = True, **kwargs):
        super().__init__(**kwargs)
        default = [round(-config.frame_x_radius), round(config.frame_x_radius), 1.0]
        self.x_range = list(x_range) if x_range is not None else default
        self.y_range = list(y_range) if y_range is not None else [round(-config.frame_y_radius),
                                                                    round(config.frame_y_radius), 1.0]
        for r in (self.x_range, self.y_range):
            if len(r) == 2:
                r.append(1.0)
        base = {"include_tip": tips, "numbers_to_exclude": [0], "exclude_origin_tick": True}
        base.update(axis_config or {})
        x_config = {**base, **(x_axis_config or {})}
        y_config = {**base, "rotation": PI / 2, "label_direction": LEFT, **(y_axis_config or {})}
        self.x_axis = self._create_axis(self.x_range, x_config, x_length)
        self.y_axis = self._create_axis(self.y_range, y_config, y_length)
        self.axes = VGroup(self.x_axis, self.y_axis)
        self.add(self.x_axis, self.y_axis)
        center = [(self.x_range[0] + self.x_range[1]) / 2, (self.y_range[0] + self.y_range[1]) / 2]
        self.shift(-self.coords_to_point(*center))

    @staticmethod
    def _create_axis(range_terms, config_, length):
        axis = NumberLine(range_terms, length=length, **config_)
        axis.shift(-axis.number_to_point(_origin_shift(range_terms[0], range_terms[1])))
        return axis

    def get_origin(self):
        return self.coords_to_point(0, 0)

    def coords_to_point(self, *coords):
        coords = [np.asarray(c, dtype=float) for c in coords]
        if len(coords) == 1 and coords[0].ndim and coords[0].shape[-1] in (2, 3):
            coords = [coords[0][..., 0], coords[0][..., 1]]
        x, y = coords[0], coords[1]
        # the axes cross where the y axis has its origin (0, or its nearest end)
        oy = _origin_shift(self.y_range[0], self.y_range[1])
        return self.x_axis.number_to_point(x) + self.y_axis.number_to_point(y) - self.y_axis.number_to_point(oy)

    c2p = coords_to_point

    def point_to_coords(self, point):
        return np.array([self.x_axis.point_to_number(point), self.y_axis.point_to_number(point)])

    p2c = point_to_coords

    def get_x_axis(self):
        return self.x_axis

    def get_y_axis(self):
        return self.y_axis

    def get_axes(self):
        return self.axes

    def add_coordinates(self, *axes_numbers, **kwargs):
        if not axes_numbers:
            axes_numbers = [None, None]
        for axis, values in zip(self.axes, axes_numbers):
            if isinstance(values, dict):
                axis.add_labels(values, **kwargs)
            else:
                axis.add_numbers(values, **kwargs)
        return self

    def _label(self, label):
        if isinstance(label, Mobject):
            return label
        from .text import text
        return text(label if "$" in label else f"${label}$", size=36)

    def get_x_axis_label(self, label, edge=UR, direction=UR, buff: float = SMALL_BUFF, **kwargs):
        return self._label(label).next_to(self.x_axis.get_critical_point(edge), direction, buff=buff)

    def get_y_axis_label(self, label, edge=UR, direction=UP * 0.5 + RIGHT, buff: float = SMALL_BUFF, **kwargs):
        return self._label(label).next_to(self.y_axis.get_critical_point(edge), direction, buff=buff)

    def get_axis_labels(self, x_label="x", y_label="y"):
        self.axis_labels = VGroup(self.get_x_axis_label(x_label), self.get_y_axis_label(y_label))
        return self.axis_labels

    def plot(self, function, x_range=None, color=None, **kwargs):
        t_range = np.array(self.x_range, dtype=float)
        if x_range is not None:
            t_range[: len(x_range)] = x_range
        graph = ParametricFunction(lambda t: self.coords_to_point(t, function(t)), t_range=t_range,
                                   color=color if color is not None else YELLOW, **kwargs)
        graph.underlying_function = function
        return graph

    def plot_parametric_curve(self, function, **kwargs):
        return ParametricFunction(lambda t: self.coords_to_point(*function(t)[:2]), **kwargs)

    def plot_line_graph(self, x_values, y_values, line_color=YELLOW, add_vertex_dots: bool = True,
                        vertex_dot_radius: float = 0.08, vertex_dot_style=None, stroke_width: float = 2, **kwargs):
        points = [self.coords_to_point(x, y) for x, y in zip(x_values, y_values)]
        graph = VGroup()
        graph.line_graph = VMobject(color=line_color, stroke_width=stroke_width, **kwargs).set_points_as_corners(points)
        graph.add(graph.line_graph)
        if add_vertex_dots:
            style = vertex_dot_style or {}
            graph.vertex_dots = VGroup(*[Dot(p, radius=vertex_dot_radius, **style) for p in points])
            graph.add(graph.vertex_dots)
        return graph

    def input_to_graph_point(self, x: float, graph):
        function = getattr(graph, "underlying_function", None)
        if function is not None:
            return self.coords_to_point(x, function(x))
        # the point of the graph's anchors at that x, interpolated
        anchors = graph.get_anchors()
        xs = np.array([self.point_to_coords(p)[0] for p in anchors])
        order = np.argsort(xs)
        y = np.interp(x, xs[order], np.array([self.point_to_coords(p)[1] for p in anchors])[order])
        return self.coords_to_point(x, y)

    i2gp = input_to_graph_point

    def get_graph_label(self, graph, label="f(x)", x_val=None, direction=RIGHT, buff: float = MED_SMALL_BUFF,
                        color=None, dot: bool = False, dot_config=None):
        mob = self._label(label)
        if x_val is None:
            x_val = self.x_range[1]
        point = self.input_to_graph_point(x_val, graph)
        mob.next_to(point, direction, buff=buff)
        mob.set_color(color if color is not None else graph.get_color())
        if dot:
            group = VGroup(Dot(point, **(dot_config or {})), mob)
            return group
        return mob

    def get_line_from_axis_to_point(self, index: int, point, line_func=DashedLine, line_config=None, color=None,
                                    stroke_width: float = 2):
        axis = self.axes[index]
        foot = axis.number_to_point(self.point_to_coords(point)[index])
        config_ = {"color": color if color is not None else theme.current().fg, "stroke_width": stroke_width}
        config_.update(line_config or {})
        return line_func(foot, point, **config_)

    def get_vertical_line(self, point, **kwargs):
        return self.get_line_from_axis_to_point(0, point, **kwargs)

    def get_horizontal_line(self, point, **kwargs):
        return self.get_line_from_axis_to_point(1, point, **kwargs)

    def get_lines_to_point(self, point, **kwargs):
        return VGroup(self.get_vertical_line(point, **kwargs), self.get_horizontal_line(point, **kwargs))

    def get_area(self, graph, x_range=None, color=(BLUE, GREEN), opacity: float = 0.3, bounded_graph=None, **kwargs):
        lo, hi = x_range if x_range is not None else (self.x_range[0], self.x_range[1])
        xs = np.linspace(lo, hi, 200)
        top = [self.input_to_graph_point(x, graph) for x in xs]
        if bounded_graph is None:
            bottom = [self.coords_to_point(x, 0) for x in xs[::-1]]
        else:
            bottom = [self.input_to_graph_point(x, bounded_graph) for x in xs[::-1]]
        color = color[0] if isinstance(color, (tuple, list)) else color
        return Polygon(*top, *bottom, fill_color=color, fill_opacity=opacity, stroke_width=0, **kwargs)


class NumberPlane(Axes):
    """Axes with a grid of background lines."""

    def __init__(self, x_range=None, y_range=None, x_length=None, y_length=None, background_line_style=None,
                 faded_line_ratio: int = 1, **kwargs):
        x_range = x_range if x_range is not None else [-config.frame_x_radius, config.frame_x_radius, 1]
        y_range = y_range if y_range is not None else [-config.frame_y_radius, config.frame_y_radius, 1]
        x_length = x_length or (x_range[1] - x_range[0])
        y_length = y_length or (y_range[1] - y_range[0])
        kwargs.setdefault("tips", False)
        axis_config = {"stroke_width": 2, "include_ticks": False, **kwargs.pop("axis_config", {})}
        super().__init__(x_range, y_range, x_length, y_length, axis_config=axis_config, **kwargs)
        style = {"stroke_color": Color("#29ABCA"), "stroke_width": 2, "stroke_opacity": 1}
        style.update(background_line_style or {})
        lines = VGroup()
        step_x, step_y = self.x_range[2] / faded_line_ratio, self.y_range[2] / faded_line_ratio
        for x in np.arange(self.x_range[0], self.x_range[1] + 1e-9, step_x):
            lines.add(Line(self.c2p(x, self.y_range[0]), self.c2p(x, self.y_range[1]), **style))
        for y in np.arange(self.y_range[0], self.y_range[1] + 1e-9, step_y):
            lines.add(Line(self.c2p(self.x_range[0], y), self.c2p(self.x_range[1], y), **style))
        self.background_lines = lines
        self.add_to_back(lines)


class ParametricFunction(VMobject):
    """The curve function(t) for t in t_range = (start, end, step), smooth through
    the samples (as manim), split at `discontinuities`."""

    def __init__(self, function, t_range=(0, 1), scaling=None, dt: float = 1e-8, discontinuities=None,
                 use_smoothing: bool = True, use_vectorized: bool = False, color=None, **kwargs):
        self.function = function
        t_range = list(t_range) + ([0.01] if len(t_range) == 2 else [])
        self.t_min, self.t_max, self.t_step = t_range
        self.dt = dt
        self.discontinuities = discontinuities
        self.use_smoothing = use_smoothing
        super().__init__(color=color if color is not None else YELLOW, **kwargs)

    def generate_points(self):
        bounds = [self.t_min]
        if self.discontinuities is not None:
            bounds += [d for d in self.discontinuities if self.t_min < d < self.t_max]
        bounds.append(self.t_max)
        points = []
        for k, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
            a = a + (self.dt if k > 0 else 0)
            b = b - (self.dt if k < len(bounds) - 2 else 0)
            ts = np.arange(a, b, self.t_step)
            ts = np.append(ts, b) if len(ts) == 0 or ts[-1] < b - 1e-12 else ts
            samples = np.array([self._point(t) for t in ts])
            part = VMobject()
            if self.use_smoothing and len(samples) > 2:
                part.set_points_smoothly(samples)
            else:
                part.set_points_as_corners(samples)
            points.append(part.points)
        self.set_points(np.concatenate(points) if points else np.zeros((0, 3)))

    def _point(self, t):
        p = np.asarray(self.function(t), dtype=float)
        return np.append(p, [0.0] * (3 - len(p)))[:3]


class FunctionGraph(ParametricFunction):
    def __init__(self, function, x_range=None, color=YELLOW, **kwargs):
        if x_range is None:
            x_range = [-config.frame_x_radius, config.frame_x_radius]
        self.underlying_function = function
        super().__init__(lambda t: np.array([t, function(t), 0.0]), t_range=x_range, color=color, **kwargs)
