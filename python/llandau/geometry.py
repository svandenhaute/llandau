"""The geometry of plots in frame units: clipping to the axes box, dashes, the paint
layers of overlapping markers, and the mobjects that stroke or fill the result."""
from __future__ import annotations

import math

import numpy as np

from .mobject import Color, VMobject, _corners

def _inside(points: np.ndarray, box) -> bool:
    """All points (at least one) finite and within box (x0, x1, y0, y1)."""
    lo, hi = points.min(0), points.max(0)  # NaN or inf: a comparison fails
    return bool(lo[0] >= box[0] and hi[0] <= box[1] and lo[1] >= box[2] and hi[1] <= box[3])


def _clip_polyline(points: np.ndarray, box) -> list[np.ndarray]:
    """Runs of a polyline inside box (x0, x1, y0, y1); NaN points break it."""
    if len(points) > 1 and _inside(points, box):
        return [points]
    x0, x1, y0, y1 = box
    a, b = points[:-1], points[1:]
    d = b - a
    # Liang-Barsky: the parameters t0..t1 of each segment inside the box
    ok = np.isfinite(a).all(1) & np.isfinite(b).all(1)
    t0, t1 = np.zeros(len(a)), np.ones(len(a))
    with np.errstate(all="ignore"):
        for p, q in ((-d[:, 0], a[:, 0] - x0), (d[:, 0], x1 - a[:, 0]),
                     (-d[:, 1], a[:, 1] - y0), (d[:, 1], y1 - a[:, 1])):
            r = q / p
            t0 = np.where(p < 0, np.maximum(t0, r), t0)
            t1 = np.where(p > 0, np.minimum(t1, r), t1)
            ok &= ~((p == 0) & (q < 0))
    ok &= t0 <= t1
    index = np.flatnonzero(ok)
    if not len(index):
        return []
    # a run goes on while the segment before ended inside, at its own end
    first = np.concatenate([[True], ~ok[index[1:] - 1] | (t1[index[1:] - 1] < 1)])
    starts, ends = a + t0[:, None] * d, a + t1[:, None] * d
    return [np.vstack([starts[run[0]], ends[run]]) for run in np.split(index, np.flatnonzero(first)[1:])]


def _clip_polygon(points: np.ndarray, box) -> np.ndarray:
    """Sutherland-Hodgman: the polygon's part inside box (x0, x1, y0, y1)."""
    x0, x1, y0, y1 = box
    poly = points[np.isfinite(points).all(1)]
    if len(poly) and _inside(poly, box):
        return poly
    for dim, value, sign in ((0, x0, 1), (0, x1, -1), (1, y0, 1), (1, y1, -1)):
        if not len(poly):
            break
        previous = np.roll(poly, 1, axis=0)
        inside = sign * (poly[:, dim] - value) >= 0
        crossing = inside != np.roll(inside, 1)
        # where an edge crosses, its crossing point comes before its vertex
        t = (value - previous[:, dim]) / np.where(crossing, poly[:, dim] - previous[:, dim], 1)
        poly = np.stack([previous + t[:, None] * (poly - previous), poly], 1)[np.column_stack([crossing, inside])]
    return poly


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


def _stroke_mob(runs, color: Color, lw: float, cap: str = "butt", join: str = "round") -> VMobject:
    """One mobject stroking the polyline runs in color (its alpha as the opacity)."""
    m = VMobject(stroke_color=color, stroke_width=lw, stroke_opacity=color.alpha, cap_style=cap, joint_type=join)
    m.stroke_color.alpha = 1
    return m.set_points(np.concatenate([_corners(r) for r in runs]))


def _fill_mob(polys, face: Color, edge=None, lw: float = 0.0, join: str = "round") -> VMobject:
    """One mobject filling the polygons with face (its alpha as the opacity), edged in edge."""
    m = VMobject(fill_color=face, fill_opacity=face.alpha, stroke_color=edge, stroke_width=lw, joint_type=join)
    m.fill_color.alpha = 1
    return m.set_points(np.concatenate([_corners(p, closed=True) for p in polys]))


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


_DASHES = {"-": None, "solid": None, "--": (3.7, 1.6), "dashed": (3.7, 1.6), ":": (1, 1.65), "dotted": (1, 1.65),
           "-.": (6.4, 1.6, 1, 1.6), "dashdot": (6.4, 1.6, 1, 1.6), "": "none", "None": "none", "none": "none"}


def _rgba(color, alpha=None) -> Color:
    c = Color(color)
    if alpha is not None:
        c.alpha = alpha
    return c
