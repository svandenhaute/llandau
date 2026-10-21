import re

import numpy as np
import pytest

from llandau import Deck, LIGHT, R, build, place, subplots


def figure():
    x = np.linspace(-1, 1, 100)
    fig, ax = subplots(figsize=(3.0, 2.0))
    ax.plot(x, np.tanh(2 * x), color="k", label="exact")
    ax.plot(x, np.tanh(2 * x) - 0.1, color="#b8b8b8", label="ODE ($A -> B$)")
    ax.scatter(x[::20], np.tanh(2 * x[::20]), s=20, color="C1", zorder=50)
    ax.set_xlabel("$lambda_(A -> B)$ [-]")
    ax.set_ylabel("coordination number [-]")
    ax.set_xlim([-1.0, 1.0])
    ax.legend(bbox_to_anchor=(0.5, 1.0))
    return fig, ax


def test_svg_is_vector_and_in_points(tmp_path):
    fig, ax = figure()
    path = fig.savefig(tmp_path / "ode.svg")
    svg = path.read_text()
    assert "<image" not in svg  # all paths: text as outlines
    width, height = (float(v) for v in re.search(r'width="([\d.]+)pt" height="([\d.]+)pt"', svg).groups())
    # tight: within the 3 x 2 inch figure, not much smaller
    assert 150 < width <= 3.1 * 72 and 100 < height <= 2.1 * 72
    # zorder: the scatter (zorder 50) comes after the lines
    assert svg.rfind('stroke="#000000" stroke-width="1.5"') < svg.rfind('fill="#E76852"')


def test_pdf_and_png(tmp_path):
    fig, _ = figure()
    pdf = fig.savefig(tmp_path / "ode.pdf")
    assert pdf.read_bytes()[:4] == b"%PDF"
    png = fig.savefig(tmp_path / "ode.png", dpi=100)
    assert png.read_bytes()[:4] == b"\x89PNG"


def test_setters_and_limits():
    fig, ax = figure()
    assert ax.get_xlim() == (-1.0, 1.0)
    lo, hi = ax.get_ylim()
    assert lo < -0.96 - 0.1 and hi > 0.96  # 5% margins around the data
    ax.set_ylim(bottom=-2)
    assert ax.get_ylim()[0] == -2


def test_figure_on_a_slide():
    class Slide(Deck):
        theme = LIGHT

        def construct(self):
            fig, _ = figure()
            self.reveal(place(fig, R.body, fit="fill"))  # lines scale with it

    deck = build([Slide])
    image = deck.render(deck.marks[0][0] - 1e-6, 320, 180)
    assert (image[..., :3].min(axis=2) < 200).mean() > 0.01


@pytest.mark.parametrize("kwargs, top_row, ylim", [
    ({}, 0, (3.5, -0.5)),                                      # rows downward, as matplotlib's
    ({"origin": "lower"}, -1, (-0.5, 3.5)),
    ({"extent": (0, 6, 0, 4)}, 0, (0, 4)),                     # row 0 at the top of the extent
    ({"extent": (0, 6, 0, 4), "origin": "lower"}, -1, (0, 4)),
    ({"extent": (6, 0, 4, 0)}, 0, (4, 0)),                     # reversed extent: reversed axes
])
def test_imshow_places_rows_as_matplotlib(kwargs, top_row, ylim):
    from llandau.mobject import ImageMobject
    Z = np.arange(24, dtype=float).reshape(4, 6)
    fig, ax = subplots(figsize=(3.0, 2.0))
    ax.imshow(Z, cmap="gray", **kwargs)
    image = next(m for m in fig.drawn() if isinstance(m, ImageMobject))
    gray = image.pixels[..., 0]
    # the top left pixel shows the top row's first column; values grow along rows and columns
    assert gray[0, 0] == gray.min() if top_row == 0 else gray[-1, 0] == gray.min()
    assert gray[0, -1] > gray[0, 0]
    assert ax.get_ylim() == pytest.approx(ylim)


def test_subplots_grid():
    fig, axes = subplots(2, 3, figsize=(6.0, 4.0))
    assert axes.shape == (2, 3)
    fig, axes = subplots(1, 3, figsize=(6.0, 2.0))
    assert axes.shape == (3,)


def test_svg_strokes_scale_with_stretching(tmp_path):
    from llandau.mobject import SVGMobject
    path = tmp_path / "ring.svg"
    path.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="600" height="300">'
                    '<circle cx="300" cy="150" r="120" fill="none" stroke="#000" stroke-width="3"/></svg>')
    scaled = SVGMobject(path, height=None).scale_to_fit_width(2.0)
    stretched = SVGMobject(path, height=None).stretch_to_fit_width(2.0)
    stretched.stretch_to_fit_height(2.0)  # the same size as scaled, in two steps
    assert stretched.submobjects[0].stroke_width == pytest.approx(scaled.submobjects[0].stroke_width)
    assert scaled.submobjects[0].stroke_width == pytest.approx(100 * 3 * 2.0 / 240, rel=1e-3)


def test_errorbar_takes_a_scalar_error():
    _, ax = subplots(figsize=(3.0, 2.0))
    bars, _ = ax.errorbar([0.0, 1.0, 2.0], [1.0, 2.0, 1.5], yerr=0.05, xerr=[0.1, 0.1, 0.1], capsize=2)
    ax._ensure()
    assert len(bars.submobjects) == 1 and bars.submobjects[0].get_num_curves() == 2 * 3 * 3  # bars and caps
    assert ax.get_ylim()[0] < 0.95 and ax.get_ylim()[1] > 2.05


def test_legend_handle_is_dashed():
    _, ax = subplots(figsize=(3.0, 2.0))
    ax.plot([0, 1], [0, 1], ls="--", label="dashed")
    ax.plot([0, 1], [1, 0], label="solid")
    legend = ax.legend()
    ax._ensure()
    rows = legend.submobjects[-1]
    dashed, solid = (rows[k][0][0] for k in range(2))
    assert len(dashed.get_subpaths()) > 1 and len(solid.get_subpaths()) == 1


def test_tick_labels_stay_with_their_ticks():
    fig, ax = subplots(figsize=(3.0, 2.0))
    ax.plot([2, 50], [1, 2])
    ax.set_xscale("log")
    ax.set_xticks([1, 10, 100], ["one", "ten", "hundred"])  # 1 and 100 are beyond the data
    ax._ensure()
    ticks, _, labels, _ = ax._ticks(0, *ax.limits[0], 1.0, 0.1)
    assert list(ticks) == [10] and labels == ["ten"]
    ax.set_xticks([1, 10], ["one"])
    with pytest.raises(ValueError):
        ax._ensure()
