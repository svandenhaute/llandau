"""matplotlib-style plots of numpy data, drawn as mobjects that can be revealed
in steps: the axes first, then groups of plot commands.

    ax = Plot(R.body.right, xlabel="time $t$ (ps)", ylabel="energy (eV)")
    md = ax.plot(t, e, label="MD")                       # a line; nothing shown yet
    with ax.group() as samples:                          # several commands, one step
        ax.scatter(ts, es, c=weights, cmap="viridis", s=30, label="samples")
        ax.fill_between(t, lo, hi, alpha=0.2)
    legend = ax.legend()

    self.reveal(ax)                                      # frame, ticks, labels
    self.reveal(md, anim=Create)                         # the line draws itself
    self.reveal(samples, legend, anim=FadeIn)

Commands take matplotlib's arguments (color, lw / linewidth, ls / linestyle,
marker, ms / markersize, alpha, label, s, c, cmap, vmin, vmax, edgecolor, ...).
Sizes are in points, as in matplotlib, times `scale` (default 2: print sizes
on a slide). Limits follow all commands made before the plot is shown, with
matplotlib's 5% margins. The style follows a thesis matplotlibrc: inward
ticks, thin black spines, markers with thin black edges, a frameless legend.
"""
from __future__ import annotations

import contextlib
import math

import numpy as np

from . import theme
from .mobject import DOWN, LEFT, ORIGIN, PI, RIGHT, UP, Color, Group, ImageMobject, VGroup, VMobject, config

__all__ = ["Plot", "COLORS", "colormap"]

# polylines keep their points to within this many points (print size)
SIMPLIFY = 0.02

# the color cycle
COLORS = ["#0074D9", "#E76852", "#63BB8D", "#A03825", "#8E6CC2", "#F2A541", "#3CB4C8", "#6F6F6F"]

# rc: sizes in points
STYLE = {
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.5,
    "lines.markersize": 8,
    "lines.markeredgewidth": 0.2,
    "xtick.major.size": 6,
    "xtick.minor.size": 3,
    "xtick.major.width": 0.4,
    "xtick.major.pad": 3.5,
    "axes.labelpad": 4.0,
    "grid.linewidth": 0.25,
    "grid.color": "#888888",
    "scatter.edgecolors": "black",
    "axes.xmargin": 0.05,
    "axes.ymargin": 0.05,
}


def colormap(name="viridis"):
    """values in [0, 1] (any shape) -> RGBA (..., 4): matplotlib's map when installed."""
    try:
        import matplotlib
        cmap = matplotlib.colormaps[name] if isinstance(name, str) else name
        return lambda v: np.asarray(cmap(np.asarray(v, dtype=float)), dtype=float)
    except (ImportError, KeyError):
        from .colormaps import apply
        return lambda v: np.concatenate([apply(v, name, normalize=False), np.ones(np.shape(v) + (1,))], -1)


# --- ticks ---------------------------------------------------------------------------

def nice_ticks(lo: float, hi: float, target: int) -> np.ndarray:
    """Ticks at 1, 2, 2.5 or 5 times a power of ten, about `target` of them in [lo, hi]."""
    if hi <= lo:
        return np.array([lo])
    raw = (hi - lo) / max(target, 1)
    power = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        step = m * power
        if (hi - lo) / step <= target:
            break
    first = math.ceil(lo / step - 1e-9) * step
    ticks = np.arange(first, hi + step * 1e-9, step)
    return np.round(ticks / step) * step


def log_ticks(lo: float, hi: float):
    """Major ticks at powers of ten in [lo, hi], minor ones at 2..9 times them."""
    a, b = math.floor(math.log10(lo) + 1e-9), math.ceil(math.log10(hi) - 1e-9)
    every = max(1, math.ceil((b - a + 1) / 8))
    major = [10.0 ** k for k in range(a, b + 1) if (k - a) % every == 0 and lo * (1 - 1e-9) <= 10.0 ** k <= hi * (1 + 1e-9)]
    minor = [m * 10.0 ** k for k in range(a - 1, b + 1) for m in range(2, 10)
             if lo <= m * 10.0 ** k <= hi] if every == 1 else []
    return np.array(major), np.array(minor)


def tick_labels(ticks) -> tuple[list[str], int]:
    """Labels for linear ticks with as few decimals as tell them apart, and the
    power of ten factored out of very large or small ones."""
    ticks = np.asarray(ticks, dtype=float)
    biggest = np.abs(ticks).max(initial=0)
    power = 0
    if biggest >= 1e4 or 0 < biggest < 1e-2:
        power = math.floor(math.log10(biggest))
        ticks = ticks / 10.0 ** power
    places = 0
    while places < 8 and np.any(np.abs(np.round(ticks, places) - ticks) > 1e-9 * max(biggest / 10.0 ** power, 1)):
        places += 1
    labels = []
    for v in ticks:
        s = f"{v:.{places}f}"
        if s.startswith("-"):
            s = "−" + s[1:] if float(s) != 0 else s[1:]
        labels.append(s)
    return labels, power


# --- geometry ----------------------------------------------------------------------

def _clip_polyline(points: np.ndarray, box) -> list[np.ndarray]:
    """Runs of a polyline inside box (x0, x1, y0, y1); NaN points break it."""
    x0, x1, y0, y1 = box
    runs, current = [], []
    for a, b in zip(points[:-1], points[1:]):
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            if len(current) > 1:
                runs.append(np.array(current))
            current = []
            continue
        # Liang-Barsky
        d = b - a
        t0, t1 = 0.0, 1.0
        ok = True
        for p, q in ((-d[0], a[0] - x0), (d[0], x1 - a[0]), (-d[1], a[1] - y0), (d[1], y1 - a[1])):
            if p == 0:
                if q < 0:
                    ok = False
                    break
            else:
                r = q / p
                if p < 0:
                    t0 = max(t0, r)
                else:
                    t1 = min(t1, r)
        if not ok or t0 > t1:
            if len(current) > 1:
                runs.append(np.array(current))
            current = []
            continue
        start, end = a + t0 * d, a + t1 * d
        if current and np.allclose(current[-1], start):
            current.append(end)
        else:
            if len(current) > 1:
                runs.append(np.array(current))
            current = [start, end]
        if t1 < 1:
            runs.append(np.array(current))
            current = []
    if len(current) > 1:
        runs.append(np.array(current))
    return runs


def _clip_polygon(points: np.ndarray, box) -> np.ndarray:
    """Sutherland-Hodgman: the polygon's part inside box (x0, x1, y0, y1)."""
    x0, x1, y0, y1 = box
    edges = [(0, x0, 1), (0, x1, -1), (1, y0, 1), (1, y1, -1)]
    poly = [p for p in points if np.all(np.isfinite(p))]
    for dim, value, sign in edges:
        if not poly:
            break
        out = []
        for i, current in enumerate(poly):
            previous = poly[i - 1]
            inside_c = sign * (current[dim] - value) >= 0
            inside_p = sign * (previous[dim] - value) >= 0
            if inside_c != inside_p:
                t = (value - previous[dim]) / (current[dim] - previous[dim])
                out.append(previous + t * (current - previous))
            if inside_c:
                out.append(current)
        poly = out
    return np.array(poly)


def _offset_polygon(poly: np.ndarray, d: float) -> np.ndarray:
    """A counterclockwise polygon with its edges moved out by d (in for d < 0), mitered."""
    e = np.roll(poly, -1, axis=0) - poly
    n = np.column_stack([e[:, 1], -e[:, 0]]) / np.hypot(e[:, 0], e[:, 1])[:, None]
    m = np.roll(n, 1, axis=0)
    return poly + d * (n + m) / (1 + (n * m).sum(axis=1))[:, None]


def _area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _layers(pts: np.ndarray, keys: list, distance: float) -> list[int]:
    """Paint layers of markers drawn in order, one mobject per (layer, key): a marker
    is above the earlier ones within distance, and shares their layer only when it
    shares their key (a stacked mobject paints its markers in turn)."""
    if len(set(keys)) < 2 or distance <= 0:
        return [0] * len(pts)
    pts = pts.tolist()
    layers, cells = [], {}
    for i, (x, y) in enumerate(pts):
        cx, cy = math.floor(x / distance), math.floor(y / distance)
        layer = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in cells.get((cx + dx, cy + dy), ()):
                    if (pts[j][0] - x) ** 2 + (pts[j][1] - y) ** 2 < distance ** 2:
                        layer = max(layer, layers[j] + (keys[j] != keys[i]))
        layers.append(layer)
        cells.setdefault((cx, cy), []).append(i)
    return layers


def simplify(points: np.ndarray, tolerance: float) -> np.ndarray:
    """Ramer-Douglas-Peucker: the polyline's points within `tolerance` of it."""
    points = np.asarray(points, dtype=float)
    if len(points) < 3 or tolerance <= 0:
        return points
    keep = np.zeros(len(points), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        p, q = points[a], points[b]
        d = q - p
        n = np.hypot(*d)
        mid = points[a + 1:b]
        dist = np.abs(d[0] * (mid[:, 1] - p[1]) - d[1] * (mid[:, 0] - p[0])) / n if n > 0 else np.hypot(*(mid - p).T)
        k = int(np.argmax(dist))
        if dist[k] > tolerance:
            keep[a + 1 + k] = True
            stack += [(a, a + 1 + k), (a + 1 + k, b)]
    return points[keep]


def _corners(points: np.ndarray, closed: bool = False) -> np.ndarray:
    """Cubic curves (4 points each) along a polyline (n, 2) -> (4(n-1), 3)."""
    p = np.asarray(points, dtype=float)
    if closed:
        p = np.vstack([p, p[:1]])
    a, b = p[:-1], p[1:]
    out = np.empty((len(a), 4, 3))
    out[..., 2] = 0
    for k, t in enumerate((0, 1 / 3, 2 / 3, 1)):
        out[:, k, :2] = a + t * (b - a)
    return out.reshape(-1, 3)


def _dashes(run: np.ndarray, pattern) -> list[np.ndarray]:
    """The dashes of a polyline run with an on-off pattern (frame units)."""
    seg = np.linalg.norm(np.diff(run, axis=0), axis=1)
    total = seg.sum()
    if total == 0:
        return []
    cum = np.concatenate([[0], np.cumsum(seg)])
    period = sum(pattern)
    out, s = [], 0.0
    while s < total:
        for k in range(0, len(pattern), 2):
            on, off = pattern[k], pattern[k + 1]
            a, b = s, min(s + on, total)
            if a >= total:
                break
            ts = np.concatenate([[a], cum[(cum > a) & (cum < b)], [b]])
            pts = np.column_stack([np.interp(ts, cum, run[:, 0]), np.interp(ts, cum, run[:, 1])])
            out.append(pts)
            s += on + off
        if period <= 0:
            break
    return out


_K = 0.5522847498  # circle by four cubics


def _marker_curves(marker: str) -> tuple[np.ndarray, bool]:
    """A marker of diameter 1 at the origin: cubic points (4 per curve, 2D), and
    whether it is filled (else stroked only)."""
    def polygon(vertices):
        return _corners(np.array(vertices), closed=True)[:, :2], True
    if marker in ("o", ".", "None"):
        r = 0.5
        quads = []
        for k in range(4):
            a = k * PI / 2
            p0 = np.array([np.cos(a), np.sin(a)]) * r
            p3 = np.array([np.cos(a + PI / 2), np.sin(a + PI / 2)]) * r
            t0 = np.array([-np.sin(a), np.cos(a)]) * r * _K
            t3 = np.array([-np.sin(a + PI / 2), np.cos(a + PI / 2)]) * r * _K
            quads.extend([p0, p0 + t0, p3 - t3, p3])
        return np.array(quads), True
    if marker == "s":
        return polygon([(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)])
    if marker == "D":
        return polygon([(0, -0.5), (0.5, 0), (0, 0.5), (-0.5, 0)])
    if marker == "d":
        return polygon([(0, -0.5), (0.3, 0), (0, 0.5), (-0.3, 0)])
    if marker in ("^", "v", "<", ">"):
        angle = {"^": PI / 2, "<": PI, "v": -PI / 2, ">": 0}[marker]
        return polygon([(0.5 * np.cos(angle + k * 2 * PI / 3), 0.5 * np.sin(angle + k * 2 * PI / 3)) for k in range(3)])
    if marker in ("p", "h", "8"):
        n = {"p": 5, "h": 6, "8": 8}[marker]
        return polygon([(0.5 * np.cos(PI / 2 + k * 2 * PI / n), 0.5 * np.sin(PI / 2 + k * 2 * PI / n)) for k in range(n)])
    if marker == "*":
        return polygon([((0.5 if k % 2 == 0 else 0.2) * np.cos(PI / 2 + k * PI / 5),
                         (0.5 if k % 2 == 0 else 0.2) * np.sin(PI / 2 + k * PI / 5)) for k in range(10)])
    if marker in ("+", "x"):
        lines = [((-0.5, 0), (0.5, 0)), ((0, -0.5), (0, 0.5))]
        if marker == "x":
            c = 0.5 / np.sqrt(2)
            lines = [((-c, -c), (c, c)), ((-c, c), (c, -c))]
        return np.concatenate([_corners(np.array(l))[:, :2] for l in lines]), False
    raise ValueError(f"unknown marker {marker!r}")


def _marker_outline(marker: str) -> list[np.ndarray]:
    """A marker of diameter 1 at the origin as polygons: its counterclockwise outline
    (filled markers), or its strokes as segments (2 points each)."""
    unit, filled = _marker_curves(marker)
    curves = unit.reshape(-1, 4, 2)
    if not filled:
        return [c[[0, 3]] for c in curves]
    t = np.linspace(0, 1, 16 if marker in ("o", ".", "None") else 1, endpoint=False)[None, :, None]
    p0, p1, p2, p3 = (curves[:, k, None] for k in range(4))
    poly = ((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3).reshape(-1, 2)
    return [poly if _area(poly) > 0 else poly[::-1]]


_DASHES = {"-": None, "solid": None, "--": (3.7, 1.6), "dashed": (3.7, 1.6), ":": (1, 1.65), "dotted": (1, 1.65),
           "-.": (6.4, 1.6, 1, 1.6), "dashdot": (6.4, 1.6, 1, 1.6), "": "none", "None": "none", "none": "none"}


def _pop(kwargs, *names, default=None):
    value = default
    for name in names:
        if name in kwargs:
            value = kwargs.pop(name)
    return value


def _rgba(color, alpha=None) -> Color:
    c = Color(color)
    if alpha is not None:
        c.alpha = alpha
    return c


# --- artists ----------------------------------------------------------------------------

class Artist(VGroup):
    """What a plot command draws; redrawn whenever the plot's layout changes."""

    # copies (e.g. by animations) belong to the same plot
    _shared_attributes = ("ax",)

    def __init__(self, ax: "Plot", kind: str, data: dict, style: dict, label=None):
        super().__init__()
        self.ax, self.kind, self.data, self.style, self.label = ax, kind, data, style, label

    def get_family(self):
        self.ax._ensure()
        return super().get_family()

    def extent(self):
        """(x values, y values, sticky x, sticky y) for autoscaling."""
        d = self.data
        return d.get("ex", []), d.get("ey", []), d.get("sticky_x", []), d.get("sticky_y", [])

    def draw(self):
        self.submobjects = []
        getattr(self.ax, f"_draw_{self.kind}")(self)
        # the parts carry the artist's z, also inside a group or a figure
        for m in self.submobjects:
            m.set_z_index(self.z_index)

    def frame_points(self) -> np.ndarray:
        """Data points in frame units (for the legend's placement)."""
        pts = self.data.get("pts")
        if pts is None:
            return np.zeros((0, 2))
        return self.ax._to_frame(*pts)


class Plot(VGroup):
    """Axes in a region, matplotlib style. The Plot itself is the frame (spines,
    ticks, labels); its commands return artists to reveal separately."""

    # copies (e.g. by animations) draw the same artists
    _shared_attributes = ("artists", "_groups", "region", "_colorbar", "_cb")

    def __init__(self, region, xlim=None, ylim=None, xlabel=None, ylabel=None, title=None, xscale: str = "linear",
                 yscale: str = "linear", grid: bool = False, xticks=None, yticks=None, xticklabels=None,
                 yticklabels=None, aspect="auto", scale: float = 2.0, fontsize: float | None = None, color=None,
                 box: bool = True, ticks: str = "in", style: dict | None = None, tick_fontsize: float | None = None,
                 legend_fontsize: float | None = None, font: str | None = None, background=None):
        super().__init__()
        from .theme import SMALL
        self.region = region
        self.xlim, self.ylim = xlim, ylim
        self.xlabel, self.ylabel, self.title_text = xlabel, ylabel, title
        self.xscale, self.yscale = xscale, yscale
        self.grid_on = grid
        self.xticks, self.yticks = xticks, yticks
        self.xticklabels, self.yticklabels = xticklabels, yticklabels
        self.aspect = aspect
        self.scale_factor = scale
        self.fontsize = fontsize or SMALL
        self.tick_fontsize = tick_fontsize or 0.85 * self.fontsize
        self.legend_fontsize = legend_fontsize or 0.8 * self.fontsize
        self.font = font
        self.fg = Color(color) if color is not None else Color(theme.current().fg)
        # legend background
        self.bg = Color(background) if background is not None else Color(theme.current().bg)
        self.box_on = box
        self.tick_direction = ticks
        self.rc = {**STYLE, **(style or {})}
        self.artists: list[Artist] = []
        self.cycle = 0
        self._colorbar = None
        self._groups: list[VGroup] = []
        self._dirty = True
        self._laying_out = False

    # --- units ----------------------------------------------------------------------

    def pt(self, value: float) -> float:
        """Points (print size) in frame units."""
        return value * self.scale_factor / 72

    def lw(self, value: float) -> float:
        """A line width in points as a manim stroke width."""
        return 100 * self.pt(value)

    # --- layout -----------------------------------------------------------------------

    def get_family(self):
        self._ensure()
        return super().get_family()

    def _ensure(self):
        if self._dirty and not self._laying_out:
            self._laying_out = True
            try:
                self._layout()
            finally:
                self._laying_out = False
            self._dirty = False

    def _limits(self, dim: int):
        given = self.xlim if dim == 0 else self.ylim
        log = (self.xscale if dim == 0 else self.yscale) == "log"
        values, sticky = [], []
        for a in self.artists:
            ex = a.extent()
            values.append(np.asarray(ex[dim], dtype=float).ravel())
            sticky.extend(np.asarray(ex[2 + dim], dtype=float).ravel())
        values = np.concatenate(values) if values else np.zeros(0)
        values = values[np.isfinite(values)]
        if log:
            values = values[values > 0]
        if given is not None and given[0] is not None and given[1] is not None:
            return float(given[0]), float(given[1])
        if len(values) == 0:
            lo, hi = (1.0, 10.0) if log else (0.0, 1.0)
        else:
            lo, hi = float(values.min()), float(values.max())
            if lo == hi:
                lo, hi = (lo / 2, hi * 2) if log else (lo - 0.5, hi + 0.5)
            margin = self.rc["axes.xmargin" if dim == 0 else "axes.ymargin"]
            if log:
                a, b = math.log10(lo), math.log10(hi)
                new_lo, new_hi = 10 ** (a - margin * (b - a)), 10 ** (b + margin * (b - a))
            else:
                new_lo, new_hi = lo - margin * (hi - lo), hi + margin * (hi - lo)
            # sticky edges (bars at 0, images): margins stop there
            for s in sticky:
                if new_lo < s <= lo:
                    new_lo = s
                if hi <= s < new_hi:
                    new_hi = s
            lo, hi = new_lo, new_hi
        if given is not None:
            lo = float(given[0]) if given[0] is not None else lo
            hi = float(given[1]) if given[1] is not None else hi
        elif any(a.data.get("inverted", (False, False))[dim] for a in self.artists):
            lo, hi = hi, lo  # an image's reversed extent, as matplotlib's
        return lo, hi

    def _ticks(self, dim: int, lo: float, hi: float, length: float, font: float):
        given = self.xticks if dim == 0 else self.yticks
        names = self.xticklabels if dim == 0 else self.yticklabels
        log = (self.xscale if dim == 0 else self.yscale) == "log"
        minor = np.zeros(0)
        if given is not None:
            major = np.asarray(given, dtype=float)
            if names is not None and len(names) != len(major):
                raise ValueError(f"{len(names)} tick labels for {len(major)} ticks")
        elif log:
            major, minor = log_ticks(min(lo, hi), max(lo, hi))
        else:
            # matplotlib's tick space: the axis length over 3 (x) or 2 (y) label heights
            major = nice_ticks(min(lo, hi), max(lo, hi), max(3, min(9, int(length / ((3 if dim == 0 else 2) * font)))))
        shown = (major >= min(lo, hi) - 1e-9 * abs(hi - lo)) & (major <= max(lo, hi) + 1e-9 * abs(hi - lo))
        major = major[shown]
        power = 0
        if names is not None:
            labels = [name for name, keep in zip(names, shown) if keep]  # with their ticks
        elif log:
            labels = [f"$10^({int(round(math.log10(v)))})$".replace("^(-", "^(−") for v in major]
        else:
            labels, power = tick_labels(major)
        return major, minor, labels, power

    def _layout(self):
        from .shapes import Line, Rectangle
        from .text import text, texts
        rc = self.rc
        r = self.region
        xlim, ylim = self._limits(0), self._limits(1)
        tick_size = self.tick_fontsize
        # ticks need lengths; first guess the box
        font = tick_size / 72
        xt, xminor, xlabels, xpower = self._ticks(0, *xlim, r.width * 0.85, font)
        yt, yminor, ylabels, ypower = self._ticks(1, *ylim, r.height * 0.8, font)
        labels = texts(xlabels + ylabels, size=tick_size, color=self.fg, font=self.font)
        xl, yl = labels[: len(xlabels)], labels[len(xlabels):]
        pad = self.pt(rc["xtick.major.pad"])
        labelpad = self.pt(rc["axes.labelpad"])
        out_tick = self.pt(rc["xtick.major.size"]) if self.tick_direction == "out" else 0.0
        x_title = self._text(self.xlabel) if self.xlabel else None
        y_title = self._text(self.ylabel).rotate(PI / 2) if self.ylabel else None
        title = self._text(self.title_text) if self.title_text else None
        # matplotlib spaces text by its font box: above the glyphs the ascent, below them the descent
        ascent, descent = 0.15 * tick_size / 72, 0.25 * tick_size / 72
        label_ascent, label_descent = 0.15 * self.fontsize / 72, 0.25 * self.fontsize / 72
        tick_h = max((m.height for m in xl), default=0) + ascent + descent
        tick_w = max((m.width for m in yl), default=0)
        bottom = out_tick + pad + tick_h + (labelpad + label_ascent + x_title.height if x_title else 0)
        left = out_tick + pad + tick_w + (labelpad + label_descent + y_title.width if y_title else 0)
        top = (title.height + 2 * labelpad + label_descent) if title else max((m.height for m in yl), default=0) / 2
        right = max(xl[-1].width / 2 if xl else 0, self.pt(2))
        if self._colorbar is not None:
            right = self._colorbar_parts()
        x0, x1 = r.x0 + left, r.x1 - right
        y0, y1 = r.y0 + bottom, r.y1 - top
        if self.aspect == "equal":
            fx = lambda v: math.log10(v) if self.xscale == "log" else v  # noqa: E731
            fy = lambda v: math.log10(v) if self.yscale == "log" else v  # noqa: E731
            sx = (x1 - x0) / (fx(xlim[1]) - fx(xlim[0]))
            sy = (y1 - y0) / (fy(ylim[1]) - fy(ylim[0]))
            s = min(sx, sy)
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            w, h = s * (fx(xlim[1]) - fx(xlim[0])), s * (fy(ylim[1]) - fy(ylim[0]))
            x0, x1, y0, y1 = cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2
        self.box = (x0, x1, y0, y1)
        self.limits = (xlim, ylim)
        # data stops at the inner edge of the spines, which it never covers
        inset = self.pt(rc["axes.linewidth"]) / 2
        self.clip = (x0 + inset, x1 - inset, y0 + inset, y1 - inset)

        self.submobjects = []
        fg, lw_axes = self.fg, self.lw(rc["axes.linewidth"])
        tick_len, minor_len = self.pt(rc["xtick.major.size"]), self.pt(rc["xtick.minor.size"])
        tick_lw = self.lw(rc["xtick.major.width"])
        sign = 1 if self.tick_direction == "in" else -1
        frame = VGroup()
        if self.grid_on:
            grid = VGroup()
            gstyle = dict(color=Color(rc["grid.color"]), stroke_width=self.lw(rc["grid.linewidth"]))
            for x in xt:
                X = self._fx(x)
                grid.add(Line([X, y0, 0], [X, y1, 0], **gstyle))
            for y in yt:
                Y = self._fy(y)
                grid.add(Line([x0, Y, 0], [x1, Y, 0], **gstyle))
            frame.add(grid)
        ticks = VGroup()
        for values, length in ((xt, tick_len), (xminor, minor_len)):
            for x in values:
                X = self._fx(x)
                ticks.add(Line([X, y0, 0], [X, y0 + sign * length, 0], color=fg, stroke_width=tick_lw))
        for values, length in ((yt, tick_len), (yminor, minor_len)):
            for y in values:
                Y = self._fy(y)
                ticks.add(Line([x0, Y, 0], [x0 + sign * length, Y, 0], color=fg, stroke_width=tick_lw))
        frame.add(ticks)
        spines = VGroup()
        corners = [[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0]]
        sides = range(4) if self.box_on else range(0, 4, 3)
        for k in sides:
            spines.add(Line(corners[k], corners[(k + 1) % 4], color=fg, stroke_width=lw_axes, cap_style="square"))
        frame.add(spines)
        # the box and its ticks sit on top of the data (below text and the legend)
        data = [a.z_index for a in self.artists if a.kind not in ("text", "colorbar", "legend")]
        VGroup(ticks, spines).set_z_index(max([2.5, *(z + 0.5 for z in data)]))
        for x, m in zip(xt, xl):
            m.move_to([self._fx(x), y0 - out_tick - pad - ascent, 0], aligned_edge=UP)
        for y, m in zip(yt, yl):
            m.move_to([x0 - out_tick - pad, self._fy(y), 0], aligned_edge=RIGHT)
        numbers = VGroup(*xl, *yl)
        for power, dim in ((xpower, 0), (ypower, 1)):
            if power:
                factor = self._text(f"$times 10^({power})$".replace("^(-", "^(−"), size=tick_size)
                if dim == 0:
                    factor.next_to([x1, y0 - out_tick - pad - tick_h, 0], DOWN, buff=pad, aligned_edge=RIGHT)  # noqa
                else:
                    factor.move_to([x0, y1 + pad, 0], aligned_edge=DOWN + LEFT)
                numbers.add(factor)
        frame.add(numbers)
        if x_title:
            x_title.move_to([(x0 + x1) / 2, y0 - out_tick - pad - tick_h - labelpad - label_ascent, 0], aligned_edge=UP)
            frame.add(x_title)
        if y_title:
            y_title.move_to([x0 - out_tick - pad - tick_w - labelpad - label_descent, (y0 + y1) / 2, 0], aligned_edge=RIGHT)
            frame.add(y_title)
        if title:
            title.move_to([(x0 + x1) / 2, y1 + labelpad + label_descent, 0], aligned_edge=DOWN)
            frame.add(title)
        self.frame = frame
        self.add(frame)
        for a in self.artists:
            a.draw()

    def _text_room(self) -> float:
        """Height the x tick labels and x label take below the box (an estimate)."""
        lines = 1 + (1 if self.xlabel else 0) + (1 if self.title_text else 0)
        return lines * 1.4 * self.fontsize / 72

    def _text(self, s: str, size: float | None = None):
        from .text import text
        return text(s, size=size or self.fontsize, color=self.fg, font=self.font)

    def _fx(self, x):
        (lo, hi), (x0, x1) = self.limits[0], self.box[:2]
        if self.xscale == "log":
            x, lo, hi = np.log10(np.maximum(x, 1e-300)), math.log10(lo), math.log10(hi)
        return x0 + (np.asarray(x, dtype=float) - lo) / (hi - lo) * (x1 - x0)

    def _fy(self, y):
        (lo, hi), (y0, y1) = self.limits[1], self.box[2:]
        if self.yscale == "log":
            y, lo, hi = np.log10(np.maximum(y, 1e-300)), math.log10(lo), math.log10(hi)
        return y0 + (np.asarray(y, dtype=float) - lo) / (hi - lo) * (y1 - y0)

    def _to_frame(self, x, y) -> np.ndarray:
        return np.column_stack([self._fx(np.asarray(x, dtype=float)), self._fy(np.asarray(y, dtype=float))])

    def coords_to_point(self, x, y):
        """A data point in frame coordinates (to place other mobjects)."""
        self._ensure()
        return np.array([float(self._fx(x)), float(self._fy(y)), 0.0])

    c2p = coords_to_point

    def point_to_coords(self, point):
        self._ensure()
        (xlo, xhi), (ylo, yhi) = self.limits
        x0, x1, y0, y1 = self.box
        fx = (point[0] - x0) / (x1 - x0)
        fy = (point[1] - y0) / (y1 - y0)
        x = 10 ** (math.log10(xlo) + fx * (math.log10(xhi) - math.log10(xlo))) if self.xscale == "log" \
            else xlo + fx * (xhi - xlo)
        y = 10 ** (math.log10(ylo) + fy * (math.log10(yhi) - math.log10(ylo))) if self.yscale == "log" \
            else ylo + fy * (yhi - ylo)
        return np.array([x, y])

    p2c = point_to_coords

    # --- commands --------------------------------------------------------------------

    def _next_color(self):
        color = COLORS[self.cycle % len(COLORS)]
        self.cycle += 1
        return color

    def _add(self, artist: Artist) -> Artist:
        zorder = artist.style.get("zorder")
        artist.z_index = _ZORDER.get(artist.kind, 2) if zorder is None else zorder
        self.artists.append(artist)
        self._dirty = True
        for g in self._groups:
            g.add(artist)
        return artist

    @contextlib.contextmanager
    def group(self):
        """Collect the artists of the commands inside into one group (one step)."""
        g = VGroup()
        self._groups.append(g)
        try:
            yield g
        finally:
            self._groups.remove(g)

    # matplotlib's setters
    def _set(self, **values):
        for key, value in values.items():
            setattr(self, key, value)
        self._dirty = True
        return self

    def set_xlim(self, left=None, right=None):
        if isinstance(left, (list, tuple, np.ndarray)):
            left, right = left
        return self._set(xlim=(left, right))

    def set_ylim(self, bottom=None, top=None):
        if isinstance(bottom, (list, tuple, np.ndarray)):
            bottom, top = bottom
        return self._set(ylim=(bottom, top))

    def get_xlim(self):
        self._ensure()
        return self.limits[0]

    def get_ylim(self):
        self._ensure()
        return self.limits[1]

    def set_xlabel(self, label: str):
        return self._set(xlabel=label)

    def set_ylabel(self, label: str):
        return self._set(ylabel=label)

    def set_title(self, title: str):
        return self._set(title_text=title)

    def set_xscale(self, scale: str):
        return self._set(xscale=scale)

    def set_yscale(self, scale: str):
        return self._set(yscale=scale)

    def set_xticks(self, ticks, labels=None):
        return self._set(xticks=ticks, xticklabels=labels)

    def set_yticks(self, ticks, labels=None):
        return self._set(yticks=ticks, yticklabels=labels)

    def set_aspect(self, aspect):
        return self._set(aspect=aspect)

    def grid(self, visible: bool = True, **kwargs):
        if "color" in kwargs:
            self.rc["grid.color"] = kwargs.pop("color")
        if "linewidth" in kwargs or "lw" in kwargs:
            self.rc["grid.linewidth"] = _pop(kwargs, "linewidth", "lw")
        return self._set(grid_on=visible)

    def plot(self, x, y=None, fmt: str | None = None, **kwargs) -> Artist:
        """A line (and markers): color, lw, ls, marker, ms, mfc, mec, mew, alpha, label, zorder."""
        if y is None:
            x, y = np.arange(len(x)), x
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        style = {"marker": None, "ls": "-"}
        if fmt:
            for m in "o.sD^v<>ph*+xd":
                if m in fmt:
                    style["marker"] = m
            for ls in ("--", "-.", ":", "-"):
                if ls in fmt:
                    style["ls"] = ls
                    break
            else:
                if style["marker"]:
                    style["ls"] = ""
            colors = [c for c in fmt if c in "bgrcmykw"]
            if colors:
                kwargs.setdefault("color", {"b": "#0074D9", "g": "#63BB8D", "r": "#E76852", "c": "#3CB4C8",
                                            "m": "#8E6CC2", "y": "#F2A541", "k": "#000000", "w": "#FFFFFF"}[colors[0]])
        color = _pop(kwargs, "color", "c", default=None) or self._next_color()
        style.update(
            color=color,
            lw=_pop(kwargs, "lw", "linewidth", default=self.rc["lines.linewidth"]),
            ls=_pop(kwargs, "ls", "linestyle", default=style["ls"]),
            marker=_pop(kwargs, "marker", default=style["marker"]),
            ms=_pop(kwargs, "ms", "markersize", default=self.rc["lines.markersize"]),
            mfc=_pop(kwargs, "mfc", "markerfacecolor", default=None),
            mec=_pop(kwargs, "mec", "markeredgecolor", default=self.rc["scatter.edgecolors"]),
            mew=_pop(kwargs, "mew", "markeredgewidth", default=self.rc["lines.markeredgewidth"]),
            alpha=_pop(kwargs, "alpha", default=None),
        )
        label = _pop(kwargs, "label")
        _unused(kwargs, "plot", style)
        data = {"x": x, "y": y, "ex": x, "ey": y, "pts": (x, y)}
        return self._add(Artist(self, "line", data, style, label))

    def _draw_line(self, a: Artist):
        s, d = a.style, a.data
        pts = self._to_frame(d["x"], d["y"])
        alpha = s["alpha"]
        color = _rgba(s["color"], alpha)
        dashes = _DASHES.get(s["ls"], None) if isinstance(s["ls"], str) else s["ls"]
        if dashes != "none":
            runs = [simplify(run, self.pt(SIMPLIFY)) for run in _clip_polyline(pts, self.clip)]
            if dashes:
                pattern = [self.pt(v * s["lw"]) for v in dashes]
                runs = [d_ for run in runs for d_ in _dashes(run, pattern)]
            if runs:
                line = VMobject(stroke_color=color, stroke_width=self.lw(s["lw"]), stroke_opacity=color.alpha,
                                joint_type="round", cap_style="butt" if dashes else "round")
                line.stroke_color.alpha = 1
                line.set_points(np.concatenate([_corners(run) for run in runs if len(run) > 1]))
                a.add(line)
        if s["marker"]:
            face = s["mfc"] if s["mfc"] is not None else s["color"]
            a.add(self._markers(pts, s["marker"], s["ms"], [face], s["mec"], s["mew"], alpha, clip=self.clip))

    def _markers(self, pts, marker, size_pt, faces, edge, edge_width, alpha, split: bool = False,
                 clip=None) -> VGroup:
        """Markers of diameter size_pt at pts; faces: one color, or one per point.
        One mobject per color (or per point with split) painting its markers in turn,
        a later marker above the earlier ones it overlaps. With clip (x0, x1, y0, y1),
        markers beyond it are left out and markers crossing it are cut to it."""
        unit, filled = _marker_curves(marker)
        if marker == ".":
            size_pt = size_pt * 0.5
        size = self.pt(size_pt)
        stroked = not filled or (edge not in (None, "none") and edge_width > 0)
        width = 0.01 * self.lw(edge_width if filled else max(edge_width, 1.0)) if stroked else 0.0
        pts = np.asarray(pts, dtype=float).reshape(-1, 2)
        faces = [faces[0]] * len(pts) if len(faces) == 1 else list(faces)
        keys = [Color(f).to_hex() + f"{Color(f).alpha:.2f}" for f in faces]
        keep = np.isfinite(pts).all(axis=1)
        cut = np.zeros(len(pts), dtype=bool)
        reach = size * np.hypot(unit[:, 0], unit[:, 1]).max() + width / 2
        if clip is not None:
            x0, x1, y0, y1 = clip
            X, Y = pts[:, 0], pts[:, 1]
            keep &= (X > x0 - reach) & (X < x1 + reach) & (Y > y0 - reach) & (Y < y1 + reach)
            cut = keep & ~((X >= x0 + reach) & (X <= x1 - reach) & (Y >= y0 + reach) & (Y <= y1 - reach))
        for i in np.flatnonzero(cut | split):
            keys[i] = int(i)  # a mobject of its own
        index = np.flatnonzero(keep)
        layers = [0] * len(index) if split else _layers(pts[index], [keys[i] for i in index], 2 * reach)
        buckets = {}
        for i, layer in zip(index, layers):
            buckets.setdefault((layer, keys[i]), []).append(i)
        group = VGroup()
        for (_, key), indices in sorted(buckets.items(), key=lambda b: (b[0][0], b[1][0])):
            face = _rgba(faces[indices[0]], alpha)
            edge_color = _rgba(edge if edge not in (None, "face") else faces[indices[0]], alpha)
            if cut[indices[0]]:
                group.add(VGroup(*self._cut_marker(pts[indices[0]], marker, size, face,
                                                   edge_color if filled else None, width, clip)))
                continue
            curves = np.concatenate([unit * size + pts[i] for i in indices])
            points = np.zeros((len(curves), 3))
            points[:, :2] = curves
            if filled:
                m = VMobject(fill_color=face, fill_opacity=face.alpha, stroke_color=edge_color,
                             stroke_width=100 * width, stroke_opacity=edge_color.alpha, joint_type="round",
                             stacked=True)
            else:
                m = VMobject(stroke_color=face, stroke_width=100 * width, stroke_opacity=face.alpha,
                             cap_style="round")
            m.fill_color.alpha = 1
            m.stroke_color.alpha = 1
            group.add(m.set_points(points))
        return group

    @staticmethod
    def _cut_marker(center, marker, size, face, edge, width, clip) -> list[VMobject]:
        """A marker crossing clip, as fills cut to it: its face, then its edge (a ring
        of the given width; edge None: the face is the stroke)."""
        def fill(color, polygons):
            polygons = [p for p in (_clip_polygon(p, clip) for p in polygons) if len(p) >= 3]
            if not polygons or color.alpha == 0:
                return []
            m = VMobject(fill_color=color, fill_opacity=color.alpha, stroke_width=0)
            m.fill_color.alpha = 1
            return [m.set_points(np.concatenate([_corners(p, closed=True) for p in polygons]))]

        parts = [p * size + center for p in _marker_outline(marker)]
        if edge is None:
            strokes = []
            for a, b in parts:
                u = (b - a) / np.hypot(*(b - a)) * width / 2
                v = np.array([-u[1], u[0]])
                strokes.append(np.array([a - v, b - v, b + v, a + v]))
            return fill(face, strokes)
        ring = []
        if width > 0:
            inner = _offset_polygon(parts[0], -width / 2)
            ring = [_offset_polygon(parts[0], width / 2)] + ([inner[::-1]] if _area(inner) > 0 else [])
        return fill(face, parts) + fill(edge, ring)

    def scatter(self, x, y, s=None, c=None, marker: str = "o", cmap="viridis", vmin=None, vmax=None,
                alpha=None, linewidths=None, edgecolors=None, label=None, split: bool = False, **kwargs) -> Artist:
        """Markers of area s (points², one or per point), colored c (a color, or values
        through cmap); split=True makes one mobject per point (to reveal them in turn)."""
        x, y = np.asarray(x, dtype=float).ravel(), np.asarray(y, dtype=float).ravel()
        color = _pop(kwargs, "color", default=None)
        linewidths = _pop(kwargs, "linewidth", "lw", default=linewidths)
        edgecolors = _pop(kwargs, "edgecolor", "ec", default=edgecolors)
        if c is None:
            c = color if color is not None else self._next_color()
        values = None
        if not isinstance(c, (str, Color)) and np.asarray(c, dtype=object).ndim == 1 and len(c) == len(x) \
                and np.issubdtype(np.asarray(c).dtype, np.number):
            values = np.asarray(c, dtype=float)
        style = dict(
            s=self.rc["lines.markersize"] ** 2 if s is None else s, c=c, values=values, marker=marker, cmap=cmap,
            vmin=vmin, vmax=vmax, alpha=alpha, split=split,
            lw=linewidths if linewidths is not None else self.rc["lines.markeredgewidth"],
            edge=edgecolors if edgecolors is not None else self.rc["scatter.edgecolors"],
        )
        _unused(kwargs, "scatter", style)
        data = {"x": x, "y": y, "ex": x, "ey": y, "pts": (x, y)}
        return self._add(Artist(self, "scatter", data, style, label))

    def _norm(self, values, vmin, vmax):
        lo = np.nanmin(values) if vmin is None else vmin
        hi = np.nanmax(values) if vmax is None else vmax
        return (values - lo) / (hi - lo) if hi > lo else np.full_like(values, 0.5), lo, hi

    def _draw_scatter(self, a: Artist):
        st, d = a.style, a.data
        pts = self._to_frame(d["x"], d["y"])
        finite = np.isfinite(pts).all(axis=1)
        if st["values"] is not None:
            norm, lo, hi = self._norm(st["values"], st["vmin"], st["vmax"])
            st["range"] = (lo, hi)
            rgba = colormap(st["cmap"])(np.clip(norm, 0, 1))
            if not st["split"]:
                rgba = np.round(rgba * 63) / 63  # one mobject per color: 64 levels per channel
            faces = [Color(v) for v in rgba[finite]]
        else:
            faces = [st["c"]]
        sizes = np.broadcast_to(np.asarray(st["s"], dtype=float), d["x"].shape)[finite]
        p = pts[finite]
        if np.allclose(sizes, sizes[0] if len(sizes) else 0):
            a.add(*self._markers(p, st["marker"], math.sqrt(sizes[0]) if len(sizes) else 1, faces, st["edge"],
                                 st["lw"], st["alpha"], st["split"], clip=self.clip))
        else:
            for i in range(len(p)):
                a.add(*self._markers(p[i:i + 1], st["marker"], math.sqrt(sizes[i]),
                                     [faces[i] if len(faces) > 1 else faces[0]], st["edge"], st["lw"], st["alpha"],
                                     clip=self.clip))

    def fill_between(self, x, y1, y2=0, color=None, alpha=None, label=None, lw: float = 0, edgecolor=None,
                     **kwargs) -> Artist:
        x = np.asarray(x, dtype=float)
        y1 = np.broadcast_to(np.asarray(y1, dtype=float), x.shape)
        y2 = np.broadcast_to(np.asarray(y2, dtype=float), x.shape)
        color = color or _pop(kwargs, "facecolor", "fc", default=None) or self._next_color()
        style = dict(color=color, alpha=alpha, lw=_pop(kwargs, "linewidth", default=lw), edge=edgecolor)
        _unused(kwargs, "fill_between", style)
        data = {"x": x, "y1": y1, "y2": y2, "ex": np.concatenate([x, x]), "ey": np.concatenate([y1, y2])}
        return self._add(Artist(self, "fill", data, style, label))

    def _draw_fill(self, a: Artist):
        st, d = a.style, a.data
        top = self._to_frame(d["x"], d["y1"])
        bottom = self._to_frame(d["x"], d["y2"])[::-1]
        poly = _clip_polygon(np.vstack([top, bottom]), self.clip)
        if len(poly) < 3:
            return
        color = _rgba(st["color"], st["alpha"] if st["alpha"] is not None else Color(st["color"]).alpha)
        edge = _rgba(st["edge"] or st["color"])
        m = VMobject(fill_color=color, fill_opacity=color.alpha, stroke_color=edge,
                     stroke_width=self.lw(st["lw"]), joint_type="round")
        m.fill_color.alpha = 1
        a.add(m.set_points(_corners(poly, closed=True)))

    def axhline(self, y: float = 0, xmin: float = 0, xmax: float = 1, **kwargs) -> Artist:
        return self._axline(1, y, xmin, xmax, **kwargs)

    def axvline(self, x: float = 0, ymin: float = 0, ymax: float = 1, **kwargs) -> Artist:
        return self._axline(0, x, ymin, ymax, **kwargs)

    def _axline(self, dim, value, lo, hi, **kwargs):
        color = _pop(kwargs, "color", "c", default=None) or self.fg
        style = dict(color=color, lw=_pop(kwargs, "lw", "linewidth", default=self.rc["lines.linewidth"]),
                     ls=_pop(kwargs, "ls", "linestyle", default="-"), alpha=_pop(kwargs, "alpha", default=None),
                     dim=dim, value=value, span=(lo, hi))
        label = _pop(kwargs, "label")
        _unused(kwargs, "axhline" if dim else "axvline", style)
        data = {"ex": [value] if dim == 0 else [], "ey": [value] if dim == 1 else []}
        return self._add(Artist(self, "axline", data, style, label))

    def _draw_axline(self, a: Artist):
        st = a.style
        x0, x1, y0, y1 = self.box
        lo, hi = st["span"]
        if st["dim"] == 1:
            Y = float(self._fy(st["value"]))
            pts = np.array([[x0 + lo * (x1 - x0), Y], [x0 + hi * (x1 - x0), Y]])
        else:
            X = float(self._fx(st["value"]))
            pts = np.array([[X, y0 + lo * (y1 - y0)], [X, y0 + hi * (y1 - y0)]])
        self._stroke(a, pts, st)

    def _stroke(self, a, pts, st):
        color = _rgba(st["color"], st["alpha"])
        dashes = _DASHES.get(st["ls"]) if isinstance(st["ls"], str) else st["ls"]
        runs = _clip_polyline(pts, self.clip)
        if dashes and dashes != "none":
            runs = [d for run in runs for d in _dashes(run, [self.pt(v * st["lw"]) for v in dashes])]
        if runs:
            m = VMobject(stroke_color=color, stroke_width=self.lw(st["lw"]), stroke_opacity=color.alpha,
                         cap_style="butt", joint_type="round")
            m.stroke_color.alpha = 1
            a.add(m.set_points(np.concatenate([_corners(r) for r in runs])))

    def bar(self, x, height, width: float = 0.8, bottom=0, color=None, alpha=None, edgecolor=None, lw: float = 0,
            label=None, align: str = "center", **kwargs) -> Artist:
        """Bars (one mobject each, e.g. to grow them in turn)."""
        x = np.asarray(x, dtype=float).ravel()
        height = np.broadcast_to(np.asarray(height, dtype=float), x.shape)
        bottom = np.broadcast_to(np.asarray(bottom, dtype=float), x.shape)
        width = np.broadcast_to(np.asarray(width, dtype=float), x.shape)
        left = x - width / 2 if align == "center" else x
        color = color or self._next_color()
        style = dict(color=color, alpha=alpha, edge=edgecolor, lw=_pop(kwargs, "linewidth", default=lw))
        _unused(kwargs, "bar", style)
        data = {"left": left, "width": width, "bottom": bottom, "height": height,
                "ex": np.concatenate([left, left + width]), "ey": np.concatenate([bottom, bottom + height]),
                "sticky_y": np.unique(bottom), "pts": (x, bottom + height)}
        return self._add(Artist(self, "bar", data, style, label))

    def _draw_bar(self, a: Artist):
        st, d = a.style, a.data
        colors = st["color"] if isinstance(st["color"], (list, tuple)) and not isinstance(st["color"][0], float) \
            else [st["color"]] * len(d["left"])
        for k, (l, w, b, h) in enumerate(zip(d["left"], d["width"], d["bottom"], d["height"])):
            corners = self._to_frame([l, l + w, l + w, l], [b, b, b + h, b + h])
            poly = _clip_polygon(corners, self.clip)
            if len(poly) < 3:
                continue
            face = _rgba(colors[k % len(colors)], st["alpha"])
            edge = _rgba(st["edge"] or colors[k % len(colors)])
            m = VMobject(fill_color=face, fill_opacity=face.alpha, stroke_color=edge,
                         stroke_width=self.lw(st["lw"]), joint_type="miter")
            m.fill_color.alpha = 1
            a.add(m.set_points(_corners(poly, closed=True)))

    def hist(self, values, bins=10, range=None, density: bool = False, histtype: str = "bar", **kwargs) -> Artist:
        counts, edges = np.histogram(np.asarray(values, dtype=float), bins=bins, range=range, density=density)
        if histtype == "step":
            xs = np.repeat(edges, 2)[1:-1]
            ys = np.repeat(counts, 2)
            return self.plot(np.concatenate([[edges[0]], xs, [edges[-1]]]), np.concatenate([[0], ys, [0]]), **kwargs)
        kwargs.setdefault("edgecolor", self.bg)
        kwargs.setdefault("lw", 0.5)
        return self.bar(edges[:-1], counts, width=np.diff(edges), align="edge", **kwargs)

    def errorbar(self, x, y, yerr=None, xerr=None, fmt: str = "o", capsize: float = 0, elinewidth=None,
                 ecolor=None, **kwargs) -> Group:
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        color = kwargs.get("color") or self._next_color()
        kwargs["color"] = color
        lw = elinewidth or kwargs.get("lw", self.rc["lines.linewidth"])
        style = dict(color=ecolor or color, lw=lw, cap=capsize, alpha=kwargs.get("alpha"), ls="-")

        def spread(err, n):
            if err is None:
                return None
            err = np.asarray(err, dtype=float)
            return (err, err) if err.ndim <= 1 else (err[0], err[1])
        data = {"x": x, "y": y, "yerr": spread(yerr, len(x)), "xerr": spread(xerr, len(x))}
        ey = [y] + ([y - data["yerr"][0], y + data["yerr"][1]] if yerr is not None else [])
        ex = [x] + ([x - data["xerr"][0], x + data["xerr"][1]] if xerr is not None else [])
        data.update(ex=np.concatenate(ex), ey=np.concatenate(ey))
        bars = self._add(Artist(self, "errorbars", data, style, None))
        points = self.plot(x, y, fmt, **kwargs) if fmt else None
        return Group(bars, points) if points is not None else Group(bars)

    def _draw_errorbars(self, a: Artist):
        st, d = a.style, a.data
        segments = []
        cap = self.pt(st["cap"]) / 2
        for err, dim in ((d["yerr"], 1), (d["xerr"], 0)):
            if err is None:
                continue
            for xi, yi, lo, hi in zip(d["x"], d["y"], err[0], err[1]):
                if dim == 1:
                    a_, b_ = self._to_frame([xi, xi], [yi - lo, yi + hi])
                else:
                    a_, b_ = self._to_frame([xi - lo, xi + hi], [yi, yi])
                segments.append(np.array([a_, b_]))
                if cap:
                    across = np.array([cap, 0]) if dim == 1 else np.array([0, cap])
                    for end in (a_, b_):
                        segments.append(np.array([end - across, end + across]))
        runs = [r for seg in segments for r in _clip_polyline(seg, self.clip)]
        if runs:
            color = _rgba(st["color"], st["alpha"])
            m = VMobject(stroke_color=color, stroke_width=self.lw(st["lw"]), stroke_opacity=color.alpha)
            m.stroke_color.alpha = 1
            a.add(m.set_points(np.concatenate([_corners(r) for r in runs])))

    def imshow(self, Z, extent=None, origin: str = "upper", cmap="viridis", vmin=None, vmax=None, alpha=None,
               **kwargs) -> Artist:
        """An image of an array (values through cmap, or RGB(A) in [0, 1] or uint8) over `extent`
        (left, right, bottom, top), as matplotlib's: column 0 at left, row 0 at top (origin
        "upper") or bottom ("lower"). The axes run as the extent does (unless their limits are
        set): without one, "upper" turns the y axis downward."""
        Z = np.asarray(Z)
        rows, cols = Z.shape[:2]
        if extent is None:
            extent = (-0.5, cols - 0.5, rows - 0.5, -0.5) if origin == "upper" else (-0.5, cols - 0.5, -0.5, rows - 0.5)
        l, r, b, t = extent
        style = dict(cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha, origin=origin)
        _unused(kwargs, "imshow", style)
        data = {"Z": Z, "extent": extent, "ex": [l, r], "ey": [b, t], "sticky_x": [l, r], "sticky_y": [b, t],
                "inverted": (r < l, t < b)}
        return self._add(Artist(self, "image", data, style, None))

    def _draw_image(self, a: Artist):
        st, d = a.style, a.data
        Z = d["Z"]
        if Z.ndim == 2:
            norm, lo, hi = self._norm(Z.astype(float), st["vmin"], st["vmax"])
            st["range"] = (lo, hi)
            rgba = colormap(st["cmap"])(np.clip(norm, 0, 1))
        else:
            rgba = Z.astype(float) / (255 if Z.dtype == np.uint8 else 1)
            if rgba.shape[2] == 3:
                rgba = np.concatenate([rgba, np.ones(rgba.shape[:2] + (1,))], 2)
        if st["alpha"] is not None:
            rgba[..., 3] *= st["alpha"]
        l, r, b, t = d["extent"]
        # where the first and last column and row land on screen, whose rows run top to bottom
        (x_first, y_first), (x_last, y_last) = self._to_frame([l, r], [t, b] if st["origin"] == "upper" else [b, t])
        if y_first < y_last:
            rgba = rgba[::-1]
        if x_first > x_last:
            rgba = rgba[:, ::-1]
        # crisp pixels under the renderer's linear filtering
        repeat = max(1, math.ceil(1024 / max(rgba.shape[:2])))
        pixels = np.repeat(np.repeat((rgba * 255).round().astype(np.uint8), repeat, 0), repeat, 1)
        image = ImageMobject(pixels)
        image.stretch_to_fit_width(abs(x_last - x_first))
        image.stretch_to_fit_height(abs(y_last - y_first))
        image.move_to([(x_first + x_last) / 2, (y_first + y_last) / 2, 0])
        a.add(image)

    def contour(self, X, Y, Z, levels=10, colors=None, cmap=None, linewidths=None, linestyles="-", **kwargs) -> Artist:
        return self._contour(X, Y, Z, levels, False, colors, cmap, linewidths, linestyles, **kwargs)

    def contourf(self, X, Y, Z, levels=10, colors=None, cmap="viridis", alpha=None, **kwargs) -> Artist:
        return self._contour(X, Y, Z, levels, True, colors, cmap, None, "-", alpha=alpha, **kwargs)

    def _contour(self, X, Y, Z, levels, filled, colors, cmap, linewidths, linestyles, alpha=None, vmin=None,
                 vmax=None, **kwargs):
        try:
            import contourpy
        except ImportError:
            raise ImportError("contour plots need contourpy (pip install contourpy)") from None
        X, Y, Z = (np.asarray(v, dtype=float) for v in (X, Y, Z))
        if X.ndim == 1:
            X, Y = np.meshgrid(X, Y)
        if np.isscalar(levels):
            levels = nice_ticks(np.nanmin(Z), np.nanmax(Z), int(levels))
            if filled:
                step = levels[1] - levels[0] if len(levels) > 1 else 1
                levels = np.concatenate([[levels[0] - step] if levels[0] > np.nanmin(Z) else [], levels,
                                         [levels[-1] + step] if levels[-1] < np.nanmax(Z) else []])
        levels = np.asarray(levels, dtype=float)
        style = dict(filled=filled, colors=colors, cmap=cmap or ("viridis" if colors is None else None), alpha=alpha,
                     lw=linewidths if linewidths is not None else self.rc["lines.linewidth"] * 0.6, ls=linestyles,
                     vmin=vmin, vmax=vmax)
        _unused(kwargs, "contour", style)
        generator = contourpy.contour_generator(X, Y, Z)
        if filled:
            parts = [generator.filled(lo, hi) for lo, hi in zip(levels[:-1], levels[1:])]
        else:
            parts = [generator.lines(level) for level in levels]
        data = {"levels": levels, "parts": parts, "ex": [X.min(), X.max()], "ey": [Y.min(), Y.max()],
                "sticky_x": [X.min(), X.max()], "sticky_y": [Y.min(), Y.max()]}
        return self._add(Artist(self, "contours", data, style, None))

    def _draw_contours(self, a: Artist):
        st, d = a.style, a.data
        levels = d["levels"]
        values = (levels[:-1] + levels[1:]) / 2 if st["filled"] else levels
        if st["colors"] is not None:
            colors = st["colors"] if isinstance(st["colors"], (list, tuple)) else [st["colors"]]
            colors = [Color(colors[k % len(colors)]) for k in range(len(values))]
        else:
            norm, lo, hi = self._norm(values, st["vmin"], st["vmax"])
            st["range"] = (lo, hi)
            colors = [Color(c) for c in colormap(st["cmap"])(np.clip(norm, 0, 1))]
        for color, part in zip(colors, d["parts"]):
            color = _rgba(color, st["alpha"])
            curves = []
            if st["filled"]:
                points, offsets = part[0], part[1]
                for pts, offs in zip(points, offsets):
                    for i0, i1 in zip(offs[:-1], offs[1:]):
                        ring = simplify(self._to_frame(pts[i0:i1, 0], pts[i0:i1, 1]), self.pt(SIMPLIFY))
                        poly = _clip_polygon(ring, self.clip)
                        if len(poly) >= 3:
                            curves.append(_corners(poly, closed=True))
                if curves:
                    m = VMobject(fill_color=color, fill_opacity=color.alpha, stroke_width=0)
                    m.fill_color.alpha = 1
                    a.add(m.set_points(np.concatenate(curves)))
            else:
                for line in part:
                    runs = [simplify(r, self.pt(SIMPLIFY))
                            for r in _clip_polyline(self._to_frame(line[:, 0], line[:, 1]), self.clip)]
                    curves.extend(_corners(r) for r in runs)
                if curves:
                    m = VMobject(stroke_color=color, stroke_width=self.lw(st["lw"]), stroke_opacity=color.alpha,
                                 joint_type="round")
                    m.stroke_color.alpha = 1
                    a.add(m.set_points(np.concatenate(curves)))

    def text(self, x, y, s: str, size: float | None = None, color=None, ha: str = "center", va: str = "center",
             transform: str = "data", zorder=None) -> Artist:
        """Text at data coordinates (transform="axes": fractions of the axes)."""
        style = dict(s=s, size=size, color=color, ha=ha, va=va, transform=transform, zorder=zorder)
        data = {"x": x, "y": y}
        return self._add(Artist(self, "text", data, style, None))

    def _draw_text(self, a: Artist):
        from .text import text
        st, d = a.style, a.data
        m = text(st["s"], size=st["size"] or self.fontsize, color=st["color"] or self.fg, font=self.font)
        if st["transform"] == "axes":
            x0, x1, y0, y1 = self.box
            point = np.array([x0 + d["x"] * (x1 - x0), y0 + d["y"] * (y1 - y0), 0])
        else:
            point = np.array([float(self._fx(d["x"])), float(self._fy(d["y"])), 0])
        edge = {"left": LEFT, "right": RIGHT}.get(st["ha"], ORIGIN) + {"top": UP, "bottom": DOWN}.get(st["va"], ORIGIN)
        a.add(m.move_to(point, aligned_edge=edge))

    def legend(self, loc: str = "best", fontsize: float | None = None, ncol: int = 1, frameon: bool = True,
               handles=None, bbox_to_anchor=None, zorder=None) -> Artist:
        """loc: "best", "upper right", ...; with bbox_to_anchor (x, y in fractions of the
        axes), the legend's loc corner sits at that point."""
        style = dict(loc=loc, fontsize=fontsize, ncol=ncol, frameon=frameon, handles=handles,
                     anchor=bbox_to_anchor, zorder=zorder)
        return self._add(Artist(self, "legend", {}, style, None))

    def _draw_legend(self, a: Artist):
        from .shapes import Rectangle
        from .text import texts
        st = a.style
        entries = [x for x in (st["handles"] or self.artists) if x.label and x.kind != "legend"]
        if not entries:
            return
        size = st["fontsize"] or self.legend_fontsize
        labels = texts([e.label for e in entries], size=size, color=self.fg, font=self.font)
        em = size / 72
        handle_w = 2 * em
        rows = VGroup()
        for e, label in zip(entries, labels):
            handle = self._handle(e, handle_w, em)
            label.next_to(handle, RIGHT, buff=0.4 * em)
            rows.add(VGroup(handle, label))
        rows.arrange_in_grid(cols=st["ncol"], buff=(0.8 * em, 0.25 * em), cell_alignment=LEFT)
        pad = 0.5 * em
        x0, x1, y0, y1 = self.box
        if st["anchor"] is not None:
            # a point in fractions of the axes: the legend's loc corner goes there
            ax_, ay_ = st["anchor"][:2]
            x0 = x1 = x0 + ax_ * (self.box[1] - self.box[0])
            y0 = y1 = y0 + ay_ * (self.box[3] - self.box[2])
        # inside the box by half a font size (matplotlib's borderaxespad)
        x0, x1, y0, y1 = x0 + pad, x1 - pad, y0 + pad, y1 - pad
        w, h = rows.width + 2 * pad, rows.height + 2 * pad
        candidates = {
            "upper right": (x1 - w / 2, y1 - h / 2), "upper left": (x0 + w / 2, y1 - h / 2),
            "lower left": (x0 + w / 2, y0 + h / 2), "lower right": (x1 - w / 2, y0 + h / 2),
            "right": (x1 - w / 2, (y0 + y1) / 2), "center left": (x0 + w / 2, (y0 + y1) / 2),
            "center right": (x1 - w / 2, (y0 + y1) / 2), "lower center": ((x0 + x1) / 2, y0 + h / 2),
            "upper center": ((x0 + x1) / 2, y1 - h / 2), "center": ((x0 + x1) / 2, (y0 + y1) / 2),
        }
        loc = st["loc"]
        if loc == "best":
            if st["anchor"] is not None:
                bx0, bx1, by0, by1 = self.box
                inside = {k: c for k, c in candidates.items() if c[0] - w / 2 >= bx0 - 1e-9 and c[0] + w / 2 <= bx1 + 1e-9
                          and c[1] - h / 2 >= by0 - 1e-9 and c[1] + h / 2 <= by1 + 1e-9}
                candidates = inside or candidates
            pts = [x.frame_points() for x in self.artists if x.kind != "legend"]
            pts = np.concatenate(pts) if pts else np.zeros((0, 2))
            pts = pts[np.isfinite(pts).all(axis=1)]

            def covered(c):
                return np.sum((np.abs(pts[:, 0] - c[0]) < w / 2) & (np.abs(pts[:, 1] - c[1]) < h / 2))
            loc = min(candidates, key=lambda k: covered(candidates[k]))
        center = np.array([*candidates[loc], 0.0])
        rows.move_to(center)
        if st["frameon"]:
            back = Rectangle(width=w, height=h, stroke_width=0, fill_color=self.bg, fill_opacity=1).move_to(center)
            a.add(back)
        a.add(rows)

    def _handle(self, e: Artist, width: float, em: float) -> VMobject:
        from .shapes import Rectangle
        s = e.style
        if e.kind in ("line", "axline"):
            group = VGroup()
            if s.get("ls") not in ("", "None", "none"):
                group.add(VMobject(stroke_color=_rgba(s["color"]), stroke_width=self.lw(s["lw"]),
                                   joint_type="round").set_points_as_corners([[0, 0, 0], [width, 0, 0]]))
            if s.get("marker"):
                face = s["mfc"] if s.get("mfc") is not None else s["color"]
                group.add(*self._markers(np.array([[width / 2, 0]]), s["marker"], s["ms"], [face], s["mec"], s["mew"],
                                         s.get("alpha")))
            if not group.submobjects:
                group.add(VMobject().set_points([[0, 0, 0]] * 4))
            return group
        if e.kind == "scatter":
            face = s["c"] if s["values"] is None else colormap(s["cmap"])(0.6)
            size = min(math.sqrt(np.max(s["s"])), 10)
            return VGroup(*self._markers(np.array([[width / 2, 0]]), s["marker"], size, [face], s["edge"], s["lw"],
                                         s["alpha"]))
        color = s.get("color", self.fg)
        color = color[0] if isinstance(color, (list, tuple)) and not isinstance(color[0], float) else color
        face = _rgba(color, s.get("alpha"))
        rect = Rectangle(width=width * 0.8, height=0.7 * em, stroke_width=0, fill_color=face, fill_opacity=face.alpha)
        rect.fill_color.alpha = 1
        return rect.shift(np.array([width / 2, 0, 0]))

    def colorbar(self, mappable: Artist, label: str | None = None, width: float = 8, ticks=None, format=None,
                 pad: float = 6, labelpad: float | None = None) -> Artist:
        """A color scale right of the plot for a scatter, image or contour artist. width, pad
        (from the axes) and labelpad (from the tick labels) in points; format: e.g. "%.1f"
        or a function of the value."""
        labelpad = self.rc["axes.labelpad"] if labelpad is None else labelpad
        style = dict(mappable=mappable, label=label, width=width, ticks=ticks, format=format, pad=pad,
                     labelpad=labelpad)
        artist = Artist(self, "colorbar", {}, style, None)
        self._colorbar = artist
        self._dirty = True
        return self._add(artist)

    def _mappable_range(self, m: Artist):
        st, d = m.style, m.data
        if m.kind == "contours":
            levels = d["levels"]
            return (st["vmin"] if st["vmin"] is not None else levels[0],
                    st["vmax"] if st["vmax"] is not None else levels[-1])
        values = st.get("values") if m.kind == "scatter" else d.get("Z")
        if values is None or np.asarray(values).ndim > 2:
            return 0.0, 1.0
        values = np.asarray(values, dtype=float)
        lo = np.nanmin(values) if st.get("vmin") is None else st["vmin"]
        hi = np.nanmax(values) if st.get("vmax") is None else st["vmax"]
        return float(lo), float(hi)

    def _colorbar_parts(self) -> float:
        """Typeset the colorbar's labels; returns the room it needs right of the box."""
        st = self._colorbar.style
        lo, hi = self._mappable_range(st["mappable"])
        if st["ticks"] is not None:
            ticks = np.asarray(st["ticks"], dtype=float)
        else:
            height = (self.region.height - self._text_room()) * 0.8
            ticks = nice_ticks(lo, hi, max(3, min(9, int(height / (2 * self.tick_fontsize / 72)))))
        ticks = ticks[(ticks >= min(lo, hi) - 1e-9) & (ticks <= max(lo, hi) + 1e-9)]
        fmt = st["format"]
        if fmt is None:
            names = tick_labels(ticks)[0]
        else:
            names = [fmt(v) if callable(fmt) else fmt % v for v in ticks]
            names = ["−" + n[1:] if n.startswith("-") else n for n in names]
        from .text import texts
        labels = texts(names, size=self.tick_fontsize, color=self.fg, font=self.font)
        title = self._text(st["label"]).rotate(-PI / 2) if st["label"] else None
        self._cb = dict(range=(lo, hi), ticks=ticks, labels=labels, title=title)
        pad, labelpad = self.pt(self.rc["xtick.major.pad"]), self.pt(st["labelpad"])
        descent = 0.25 * self.fontsize / 72
        return (self.pt(st["pad"]) + self.pt(st["width"]) + pad + max((l.width for l in labels), default=0)
                + (labelpad + descent + title.width if title else 0) + self.pt(2))

    def _draw_colorbar(self, a: Artist):
        from .shapes import Line, Rectangle
        st, cb = a.style, self._cb
        m = st["mappable"]
        lo, hi = cb["range"]
        x0, x1, y0, y1 = self.box
        left, width = x1 + self.pt(st["pad"]), self.pt(st["width"])
        right = left + width
        ys = lambda v: y0 + (v - lo) / (hi - lo) * (y1 - y0) if hi > lo else (y0 + y1) / 2  # noqa: E731
        cmap = m.style.get("cmap") or "viridis"
        if m.kind == "contours" and m.style["filled"]:
            # one band per level, as contourf
            levels = m.data["levels"]
            middles = (levels[:-1] + levels[1:]) / 2
            colors = colormap(cmap)(np.clip((middles - lo) / (hi - lo), 0, 1)) if hi > lo else colormap(cmap)(middles * 0)
            alpha = m.style.get("alpha")
            for (a_, b_), c in zip(zip(levels[:-1], levels[1:]), colors):
                band = Rectangle(width=width, height=max(ys(b_) - ys(a_), 0), stroke_width=0,
                                 fill_color=Color(c[:3]), fill_opacity=1 if alpha is None else alpha)
                a.add(band.move_to([left + width / 2, (ys(a_) + ys(b_)) / 2, 0]))
        else:
            ramp = colormap(cmap)(np.linspace(1, 0, 256))[:, None, :].repeat(8, axis=1)
            image = ImageMobject((ramp * 255).round().astype(np.uint8), opacity=m.style.get("alpha") or 1.0)
            image.stretch_to_fit_width(width).stretch_to_fit_height(y1 - y0)
            a.add(image.move_to([left + width / 2, (y0 + y1) / 2, 0]))
        a.add(Rectangle(width=width, height=y1 - y0, color=self.fg, stroke_width=self.lw(self.rc["axes.linewidth"]))
              .move_to([left + width / 2, (y0 + y1) / 2, 0]))
        pad = self.pt(self.rc["xtick.major.pad"])
        tick_len = self.pt(self.rc["xtick.major.size"]) * 0.6
        for v, label in zip(cb["ticks"], cb["labels"]):
            Y = ys(v)
            a.add(Line([right, Y, 0], [right - tick_len, Y, 0], color=self.fg,
                       stroke_width=self.lw(self.rc["xtick.major.width"])))
            a.add(label.move_to([right + pad, Y, 0], aligned_edge=LEFT))
        if cb["title"] is not None:
            edge = max((l.get_right()[0] for l in cb["labels"]), default=right)
            gap = self.pt(st["labelpad"]) + 0.25 * self.fontsize / 72  # the label's descent faces the ticks
            a.add(cb["title"].move_to([edge + gap, (y0 + y1) / 2, 0], aligned_edge=LEFT))


# matplotlib's default drawing order
_ZORDER = {"image": 0, "contours": 1, "fill": 1, "bar": 1, "scatter": 1, "line": 2, "axline": 2, "errorbars": 2,
           "text": 3, "colorbar": 3, "legend": 5}


def _unused(kwargs, command, style):
    """Takes zorder into style; fails on arguments the command does not know."""
    style["zorder"] = kwargs.pop("zorder", None)
    for ignored in ("rasterized", "clip_on", "solid_capstyle", "antialiased"):
        kwargs.pop(ignored, None)
    if kwargs:
        raise TypeError(f"{command}() got unknown arguments: {', '.join(sorted(kwargs))}")
