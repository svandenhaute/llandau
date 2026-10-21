"""Marker shapes (matplotlib's names) and the mobjects painting them: one per color,
each marker over the earlier ones it overlaps, cut to the axes box at its edges."""
from __future__ import annotations

from functools import cache

import numpy as np

from .geometry import _area, _clip_polygon, _fill_mob, _layers, _offset_polygon, _rgba
from .mobject import PI, VGroup, VMobject, _corners

_K = 0.5522847498  # circle by four cubics


@cache
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


def markers(pts, marker, size_pt, face, edge, edge_width, alpha, pt: float, split: bool = False,
            clip=None) -> VGroup:
    """Markers of diameter size_pt points (one, or one per point) at pts, with pt frame
    units per point; face: one color, or RGBA (n, 4), one per point. One mobject per
    color (or per point with split) painting its markers in turn, a later marker above
    the earlier ones it overlaps. With clip (x0, x1, y0, y1), markers beyond it are
    left out and markers crossing it are cut to it."""
    unit, filled = _marker_curves(marker)
    if marker == ".":
        size_pt = size_pt * 0.5
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    size = np.broadcast_to(pt * np.asarray(size_pt, dtype=float), len(pts))
    stroked = not filled or (edge not in (None, "none") and edge_width > 0)
    width = pt * (edge_width if filled else max(edge_width, 1.0)) if stroked else 0.0
    per_point = isinstance(face, np.ndarray) and face.ndim == 2
    keys = list(map(bytes, (face * 255).round().astype(np.uint8))) if per_point else [None] * len(pts)
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
    layers = [0] * len(index) if split else _layers(pts[index], [keys[i] for i in index], 2 * reach.max(initial=0))
    buckets = {}
    for i, layer in zip(index, layers):
        buckets.setdefault((layer, keys[i]), []).append(i)
    group = VGroup()
    for (_, key), indices in sorted(buckets.items(), key=lambda b: (b[0][0], b[1][0])):
        first = indices[0]
        own = face[first] if per_point else face
        face_color = _rgba(own, alpha)
        edge_color = _rgba(edge if edge not in (None, "face") else own, alpha)
        if cut[first]:
            group.add(VGroup(*_cut_marker(pts[first], marker, size[first], face_color,
                                          edge_color if filled else None, width, clip)))
            continue
        points = np.zeros((len(indices) * len(unit), 3))
        points[:, :2] = (unit[None] * size[indices, None, None] + pts[indices, None]).reshape(-1, 2)
        if filled:
            m = VMobject(fill_color=face_color, fill_opacity=face_color.alpha, stroke_color=edge_color,
                         stroke_width=100 * width, stroke_opacity=edge_color.alpha, joint_type="round",
                         stacked=True)
        else:
            m = VMobject(stroke_color=face_color, stroke_width=100 * width, stroke_opacity=face_color.alpha,
                         cap_style="round")
        m.fill_color.alpha = m.stroke_color.alpha = 1
        group.add(m.set_points(points))
    return group

def _cut_marker(center, marker, size, face, edge, width, clip) -> list[VMobject]:
    """A marker crossing clip, as fills cut to it: its face, then its edge (a ring
    of the given width; edge None: the face is the stroke)."""
    def fill(color, polygons):
        polygons = [p for p in (_clip_polygon(p, clip) for p in polygons) if len(p) >= 3]
        return [_fill_mob(polygons, color)] if polygons and color.alpha != 0 else []

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
