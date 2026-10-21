import numpy as np
import pytest

from llandau import (
    BLACK, BLUE, UP, Create, Deck, Dot, FadeIn, FadeOut, LaggedStart, LIGHT, R, Rectangle, Scene, Square, Transform,
    Unwrite, VGroup, Write, build, place, text,
)
from llandau.mobject import smooth_handles


def ink(image) -> float:
    """Fraction of pixels that are not the white background."""
    return float((image[..., :3].min(axis=2) < 200).mean())


class Shapes(Scene):
    def construct(self):
        self.camera.background_color = "#FFFFFF"
        square = Square(color=BLUE, fill_opacity=1)
        self.play(FadeIn(square))
        self.next_slide()
        self.play(Transform(square, Dot(radius=0.5, color=BLACK)))
        self.next_slide()
        self.play(FadeOut(square))


def test_timeline_follows_the_animations():
    deck = build([Shapes])
    marks = [t for t, _, _ in deck.marks]
    assert len(marks) == 3
    assert marks == pytest.approx([1.0, 2.0, 3.0])
    assert ink(deck.render(0.0, 160, 90)) < 0.001  # FadeIn starts transparent
    square = ink(deck.render(0.999, 160, 90))
    dot = ink(deck.render(1.999, 160, 90))
    assert square > 0.03 and 0.005 < dot < square  # a 2 x 2 square is 3.5% of the frame
    assert ink(deck.render(2.999, 160, 90)) < 0.001  # faded out


class Steps(Deck):
    theme = LIGHT

    def construct(self):
        self.title("steps")
        self.reveal(place(Rectangle(fill_opacity=1, color=BLACK), R.body.left))
        self.reveal(place(text("written"), R.body.right), anim=Write)


def test_deck_steps_and_text():
    deck = build([Steps])
    assert len(deck.marks) == 2  # nothing follows the last step
    first, second = (deck.render(t - 1e-6, 320, 180) for t, _, _ in deck.marks)
    assert ink(second) > ink(first) > 0
    # halfway through Write the text is partly there
    middle = deck.render(deck.marks[0][0] + 0.2, 320, 180)
    assert ink(first) < ink(middle) < ink(second)


def test_lagged_writes_overlap():
    class Lagged(Scene):
        def construct(self):
            dots = VGroup(*[Dot([x, 0, 0]) for x in range(-3, 4)])
            self.play(LaggedStart(*[Create(d) for d in dots], lag_ratio=0.5))

    deck = build([Lagged])
    assert deck.duration == pytest.approx(4.0)


def test_unwrite_honors_a_custom_outline():
    # the native write clip draws manim's default outline (width 2, the stroke color):
    # any other outline is baked, and drawn
    def unwrite(**kwargs):
        class Unwriting(Scene):
            def setup(self):
                self.camera.background_color = "#FFFFFF"  # the deck's: no backdrop clip

            def construct(self):
                square = Square(color=BLACK, fill_opacity=1, stroke_width=0)
                self.add(square)
                self.play(Unwrite(square, **kwargs))

        deck = build([Unwriting])
        return deck.num_clips, ink(deck.render(0.75, 320, 180))  # half of the outline is left

    native, thin = unwrite()
    baked, thick = unwrite(stroke_width=20)
    assert native == 1 and baked > 10
    assert 0 < thin and 3 * thin < thick


def test_updaters_are_baked():
    from llandau import ValueTracker, always_redraw

    class Tracking(Scene):
        def construct(self):
            self.camera.background_color = "#FFFFFF"
            x = ValueTracker(-2)
            dot = always_redraw(lambda: Dot([x.get_value(), 0, 0], radius=0.3, color=BLACK))
            self.add(dot)
            self.wait(0.5)  # still: the updater redraws the same dot
            self.play(x.animate.set_value(2), run_time=1)  # moving: sampled at FPS
            self.next_slide()
            self.wait(0.5)

    deck = build([Tracking])
    assert deck.num_clips > 30
    assert deck.marks[0][0] == pytest.approx(1.5)
    left, right = (deck.render(t, 160, 90)[..., :3].min(axis=2) < 200 for t in (0.25, 1.49))
    assert left[:, :80].sum() > 0 and left[:, 80:].sum() == 0  # the dot is on the left, then on the right
    assert right[:, 80:].sum() > 0 and right[:, :80].sum() == 0


def test_spline_handles_are_manims():
    anchors = np.array([[0, 0, 0], [1, 2, 0], [3, 1, 0], [4, 4, 0]], float)
    h1, h2 = smooth_handles(anchors)
    # C1 and C2 continuity at the inner anchors
    for i in (1, 2):
        assert np.allclose(h2[i - 1] + h1[i], 2 * anchors[i])
        assert np.allclose(h1[i - 1] - 2 * h2[i - 1], h2[i] - 2 * h1[i])


def test_plot_steps_and_ticks():
    from llandau import Plot
    from llandau.plot import nice_ticks, tick_labels

    assert list(nice_ticks(0, 10, 5)) == [0, 2, 4, 6, 8, 10]
    assert tick_labels([-0.5, 0, 0.25]) == (["−0.50", "0.00", "0.25"], 0)
    assert tick_labels([0, 20000, 40000])[1] == 4

    class Plotting(Deck):
        theme = LIGHT

        def construct(self):
            x = np.linspace(0, 1, 50)
            ax = Plot(R.body, xlabel="$x$", ylabel="y")
            line = ax.plot(x, x ** 2, label="square")
            with ax.group() as points:
                ax.scatter(x[::5], x[::5], c=x[::5], s=30)
                ax.fill_between(x, 0, x, alpha=0.2)
            legend = ax.legend()
            self.reveal(ax)
            self.reveal(line, anim=Create)
            self.reveal(points, legend, anim=FadeIn)

    deck = build([Plotting])
    assert len(deck.marks) == 3
    frame, line, everything = (ink(deck.render(t - 1e-6, 320, 180)) for t, _, _ in deck.marks)
    assert 0 < frame < line < everything


def test_axes_like_manim():
    from llandau import Axes

    ax = Axes(x_range=[-1, 5, 1], y_range=[-2, 4, 1])
    assert np.allclose(ax.c2p(0, 0), [-4, -1, 0])  # manim's values
    assert np.allclose(ax.c2p(5, 4), [6, 3, 0])
    ax.shift(UP).scale(0.5)
    assert np.allclose(ax.c2p(3, 1), [1, 1, 0])
    graph = ax.plot(lambda x: x ** 2 / 10, x_range=[0, 4])
    assert np.allclose(graph.get_end(), ax.c2p(4, 1.6))


def test_live_view_spans_its_step():
    from llandau import live_scene

    @live_scene("dot")
    def dot(viewer):
        viewer.load(np.zeros((2, 1, 3), np.float32) + [[[0, 0, 0]], [[1, 0, 0]]], np.array([6]))

    class Live(Deck):
        theme = LIGHT

        def construct(self):
            self.reveal(text("before"))
            self.live("dot", R.body.right, still=False, frame=1)
            self.reveal(text("during"))
            self.reveal(text("after"))

    deck = build([Live])
    assert len(deck.marks) == 3
