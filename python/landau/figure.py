"""Figures for papers: plots saved as SVG, PDF or PNG, matplotlib style.

    from landau import subplots

    fig, ax = subplots(figsize=(3.0, 2.0))            # inches
    ax.plot(x, y, color="k", label="exact")
    ax.scatter(xs, ys, s=20, color=BLUE, zorder=50)
    ax.set_xlabel("$lambda_(A -> B)$")               # math in Typst syntax
    ax.legend()
    fig.savefig("ode.svg")                           # vector; .pdf and .png too

Sizes are print sizes: the figure in inches, lines, markers and fonts in points.
The style follows the thesis matplotlibrc (Linux Biolinum at 9.5 pt, thin black
spines, inward ticks). Text is typeset with Typst and saved as outlines, so
figures look the same wherever they are included. A figure is also a mobject:
place() it on a slide, or reveal its plots' artists in steps.
"""
from __future__ import annotations

import base64
import subprocess
import tempfile
import zlib
from pathlib import Path

import numpy as np

from .layout import Region
from .mobject import Color, Group, ImageMobject, Mobject, VMobject

__all__ = ["Figure", "subplots", "figure", "FIGURE_STYLE"]

# the thesis matplotlibrc, in points
FIGURE_STYLE = {
    "font.size": 9.5,
    "legend.fontsize": 8,
    "xtick.major.pad": 1.2,
    "axes.labelpad": 1.2,
}

_POINTS = 72  # points per inch (figure units are inches)


class Figure(Group):
    """A figure of `figsize` inches holding plots (subplots or add_plot) and any other
    mobjects (fig.add); its units are inches."""

    def __init__(self, figsize=(3.5, 2.5), fontsize: float | None = None, font: str | None = None, color="#000000",
                 style: dict | None = None):
        self.extras: list[Mobject] = []
        super().__init__()
        self.figsize = tuple(float(v) for v in figsize)
        self.style = {**FIGURE_STYLE, **(style or {})}
        self.fontsize = fontsize or self.style["font.size"]
        self.font = font
        self.color = Color(color)
        self.axes: list = []

    @property
    def region(self) -> Region:
        w, h = self.figsize
        return Region(0.0, w, 0.0, h)

    def add_plot(self, region: Region | None = None, **kwargs):
        """A plot over `region` (inches from the lower left; default: the whole figure)."""
        from .plot import Plot
        rc = {k: v for k, v in self.style.items() if "." in k and k not in ("font.size", "legend.fontsize")}
        kwargs.setdefault("style", {})
        kwargs["style"] = {**rc, **kwargs["style"]}
        ax = Plot(region if region is not None else self.region, scale=1.0, fontsize=self.fontsize,
                  tick_fontsize=self.fontsize, legend_fontsize=self.style["legend.fontsize"], font=self.font,
                  color=self.color, background="#FFFFFF", **kwargs)
        self.axes.append(ax)
        return ax

    def gca(self):
        """The last plot (a new one filling the figure if there is none)."""
        return self.axes[-1] if self.axes else self.add_plot()

    def subplots(self, nrows: int = 1, ncols: int = 1, wspace: float = 0.1, hspace: float = 0.1, **kwargs):
        """A grid of plots (rows from the top); wspace, hspace: gaps in inches."""
        cells = self.region.grid(nrows, ncols, gap=max(wspace, hspace)) if wspace == hspace else [
            cell for row in self.region.vsplit(*[1] * nrows, gap=hspace) for cell in row.hsplit(*[1] * ncols, gap=wspace)
        ]
        axes = [self.add_plot(cell, **kwargs) for cell in cells]
        if nrows == ncols == 1:
            return axes[0]
        return np.array(axes, dtype=object).reshape(nrows, ncols).squeeze()

    def add(self, *mobjects):
        """Other mobjects on the figure (in inches)."""
        self.extras.extend(mobjects)
        return self

    def scale(self, factor: float, **kwargs):
        # a figure scales as a picture: line widths too
        kwargs.setdefault("scale_stroke", True)
        return super().scale(factor, **kwargs)

    def get_family(self):
        # the plots, then everything they draw in drawing order, then the extras
        parts = []
        for ax in self.axes:
            ax._ensure()
            parts.append(ax)
        artists = sorted((a for ax in self.axes for a in ax.artists), key=lambda a: a.z_index)
        self.submobjects = parts + artists + list(self.extras)
        # the extras stay on top
        top = max((getattr(m, "z_index", 0) for a in parts + artists for m in a.get_family()), default=0)
        for m in self.extras:
            for part in m.get_family():
                part.z_index = max(getattr(part, "z_index", 0), top)
        return super().get_family()

    def drawn(self) -> list[Mobject]:
        """Everything drawn, bottom to top (by z_index, then in order)."""
        return sorted((m for m in self.get_family()[1:] if _drawable(m)), key=lambda m: getattr(m, "z_index", 0))

    # --- output ---------------------------------------------------------------------------

    def savefig(self, path, format: str | None = None, transparent: bool = True, dpi: int = 300,
                bbox_inches: str | None = "tight", pad_inches: float = 0.02):
        """Save as SVG (vector), PDF (vector) or PNG (dpi), by the suffix or `format`.
        bbox_inches="tight": cropped to what is drawn, plus pad_inches."""
        path = Path(path)
        format = (format or path.suffix.lstrip(".") or "svg").lower()
        svg = to_svg(self.drawn(), self._bounds(bbox_inches, pad_inches), None if transparent else "#FFFFFF")
        if format == "svg":
            path.write_text(svg)
        elif format in ("pdf", "png"):
            _typst_convert(svg, path, format, dpi, transparent)
        else:
            raise ValueError(f"unknown format {format!r}: svg, pdf or png")
        return path

    def _bounds(self, bbox_inches, pad):
        if bbox_inches != "tight":
            w, h = self.figsize
            return (0.0, w, 0.0, h)
        points = np.concatenate([m.get_all_points() for m in self.drawn()] or [np.zeros((1, 3))])
        lo, hi = points.min(axis=0), points.max(axis=0)
        # strokes reach half their width beyond their points
        reach = max((0.005 * m.stroke_width for m in self.drawn() if isinstance(m, VMobject)), default=0)
        return (lo[0] - pad - reach, hi[0] + pad + reach, lo[1] - pad - reach, hi[1] + pad + reach)

    def show(self, size=(1200, 800)):
        """Open the figure in a window."""
        from .scene import Scene, build
        with tempfile.TemporaryDirectory() as folder:
            png = Path(folder) / "figure.png"
            self.savefig(png, dpi=200, transparent=False)
            image = ImageMobject(str(png))

            class Preview(Scene):
                def construct(scene):
                    from .mobject import config
                    scene.camera.background_color = "#FFFFFF"
                    image.scale(min(0.95 * config.frame_width / image.width, 0.95 * config.frame_height / image.height))
                    scene.add(image)
            build([Preview], background=(1, 1, 1, 1)).play(title="figure", size=size)


def figure(figsize=(3.5, 2.5), **kwargs) -> Figure:
    return Figure(figsize, **kwargs)


def subplots(nrows: int = 1, ncols: int = 1, figsize=(3.5, 2.5), wspace: float = 0.1, hspace: float = 0.1,
             fontsize: float | None = None, font: str | None = None, style: dict | None = None, **kwargs):
    """(figure, plot) or (figure, array of plots), as matplotlib's plt.subplots."""
    fig = Figure(figsize, fontsize=fontsize, font=font, style=style)
    return fig, fig.subplots(nrows, ncols, wspace=wspace, hspace=hspace, **kwargs)


def _drawable(m) -> bool:
    if isinstance(m, ImageMobject):
        return True
    if not isinstance(m, VMobject) or len(m.points) < 4:
        return False
    fill = m.fill_opacity * m.fill_color.alpha > 0
    stroke = m.stroke_opacity * m.stroke_color.alpha > 0 and m.stroke_width > 0
    return fill or stroke


# --- SVG ----------------------------------------------------------------------------------

def _num(v: float) -> str:
    s = f"{v:.3f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def _path_data(points: np.ndarray, x0: float, y1: float) -> str:
    """SVG path data of cubic curves (4 points each), in points from the top left."""
    pts = points[: 4 * (len(points) // 4), :2]
    X = (pts[:, 0] - x0) * _POINTS
    Y = (y1 - pts[:, 1]) * _POINTS
    parts, start, last = [], None, None
    for k in range(0, len(pts), 4):
        a = (X[k], Y[k])
        if last is None or abs(a[0] - last[0]) > 1e-6 or abs(a[1] - last[1]) > 1e-6:
            if start is not None and last is not None and abs(last[0] - start[0]) < 1e-6 and abs(last[1] - start[1]) < 1e-6:
                parts.append("Z")
            parts.append(f"M{_num(a[0])} {_num(a[1])}")
            start = a
        parts.append("C" + " ".join(f"{_num(X[k + i])} {_num(Y[k + i])}" for i in (1, 2, 3)))
        last = (X[k + 3], Y[k + 3])
    if start is not None and last is not None and abs(last[0] - start[0]) < 1e-6 and abs(last[1] - start[1]) < 1e-6:
        parts.append("Z")
    return "".join(parts)


def _png(pixels: np.ndarray) -> bytes:
    """A minimal PNG encoder (RGBA, 8 bit)."""
    h, w = pixels.shape[:2]
    raw = b"".join(b"\x00" + pixels[y].tobytes() for y in range(h))

    def chunk(kind, data):
        return (len(data).to_bytes(4, "big") + kind + data
                + (zlib.crc32(kind + data) & 0xFFFFFFFF).to_bytes(4, "big"))
    header = w.to_bytes(4, "big") + h.to_bytes(4, "big") + bytes([8, 6, 0, 0, 0])
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")


def to_svg(mobjects, bounds, background: str | None = None) -> str:
    """SVG of mobjects (bottom to top) within bounds (x0, x1, y0, y1), units: inches."""
    x0, x1, y0, y1 = bounds
    width, height = (x1 - x0) * _POINTS, (y1 - y0) * _POINTS
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
           f'width="{_num(width)}pt" height="{_num(height)}pt" viewBox="0 0 {_num(width)} {_num(height)}">']
    if background:
        out.append(f'<rect width="100%" height="100%" fill="{background}"/>')
    joins, caps = {"miter": "miter", "round": "round", "bevel": "bevel"}, {"butt": "butt", "round": "round",
                                                                            "square": "square", "projecting": "square"}
    for m in mobjects:
        if isinstance(m, ImageMobject):
            lo, hi = m.points.min(axis=0), m.points.max(axis=0)
            if m.path is not None:
                data = Path(m.path).read_bytes()
                mime = "image/png" if data[:4] == b"\x89PNG" else "image/jpeg"
            else:
                data, mime = _png(m.pixels), "image/png"
            href = f"data:{mime};base64,{base64.b64encode(data).decode()}"
            opacity = f' opacity="{_num(m.opacity)}"' if m.opacity < 1 else ""
            out.append(f'<image x="{_num((lo[0] - x0) * _POINTS)}" y="{_num((y1 - hi[1]) * _POINTS)}" '
                       f'width="{_num((hi[0] - lo[0]) * _POINTS)}" height="{_num((hi[1] - lo[1]) * _POINTS)}" '
                       f'preserveAspectRatio="none"{opacity} xlink:href="{href}"/>')
            continue
        attrs = []
        fill_alpha = m.fill_opacity * m.fill_color.alpha
        if fill_alpha > 0:
            attrs.append(f'fill="{m.fill_color.to_hex()}"')
            if fill_alpha < 1:
                attrs.append(f'fill-opacity="{_num(fill_alpha)}"')
        else:
            attrs.append('fill="none"')
        stroke_alpha = m.stroke_opacity * m.stroke_color.alpha
        if stroke_alpha > 0 and m.stroke_width > 0:
            attrs.append(f'stroke="{m.stroke_color.to_hex()}" stroke-width="{_num(0.01 * m.stroke_width * _POINTS)}"')
            if stroke_alpha < 1:
                attrs.append(f'stroke-opacity="{_num(stroke_alpha)}"')
            join = joins.get(getattr(m, "joint_type", "miter"), "miter")
            cap = caps.get(getattr(m, "cap_style", "butt"), "butt")
            if join != "miter":
                attrs.append(f'stroke-linejoin="{join}"')
            else:
                attrs.append('stroke-miterlimit="10"')
            if cap != "butt":
                attrs.append(f'stroke-linecap="{cap}"')
        out.append(f'<path d="{_path_data(m.points, x0, y1)}" {" ".join(attrs)}/>')
    out.append("</svg>")
    return "\n".join(out)


def _typst_convert(svg: str, path: Path, format: str, dpi: int, transparent: bool):
    """PDF or PNG of an SVG, through Typst."""
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        (folder / "figure.svg").write_text(svg)
        fill = "none" if transparent else "white"
        (folder / "figure.typ").write_text(
            f'#set page(width: auto, height: auto, margin: 0pt, fill: {fill})\n#image("figure.svg")\n')
        command = ["typst", "compile", "--root", str(folder), str(folder / "figure.typ"), str(path.resolve())]
        if format == "png":
            command += ["--format", "png", "--ppi", str(dpi)]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"typst could not convert the figure:\n{result.stderr}")
