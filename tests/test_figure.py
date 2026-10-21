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
