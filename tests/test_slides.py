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
    middle = deck.render(deck.marks[0][0] + 0.1, 320, 180)  # reveal: run_time 0.2
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


def test_live_grid_opens():
    """Several live views on one step, each opening up to the body on a click."""
    from llandau import live_scene

    for k in range(4):
        @live_scene(f"cell{k}")
        def cell(viewer, k=k):
            viewer.load(np.zeros((1, 1, 3), np.float32), np.array([6 + k]))

    class Grid(Deck):
        theme = LIGHT

        def construct(self):
            for k, region in enumerate(R.body.grid(2, 2, gap=0.2)):
                self.live(f"cell{k}", region, still=False, opens=R.body)
            self.reveal(text("after"))

    deck = build([Grid])
    assert len(deck.marks) == 1 and deck.num_lives == 4  # the views and the text share the first step


def test_layer_step_keys():
    """Labels order as number then letter: 1 < 1a < 1b < 2 < 10 < 11, and "11" is not a sub-step of 1."""
    from llandau.media import step_key, _steps, Layer
    assert step_key("3") == (3, "")
    assert step_key("3b arrows") == (3, "b")
    assert step_key("Layer 11") == (11, "")
    assert step_key("11") == (11, "")
    assert step_key("1A") == (1, "a")
    assert step_key("arrows 3") is None
    assert step_key("3abc") is None
    labels = ["11", "2", "1b", "10", "1", "1a", "_guide"]
    steps = _steps([Layer(f"id{i}", label) for i, label in enumerate(labels)], "label", None)
    assert [[l.label for l in s] for s in steps] == [["1"], ["1a"], ["1b"], ["2"], ["10"], ["11"]]
    shared = _steps([Layer("a", "2 boxes"), Layer("b", "2 labels"), Layer("c", "1")], "label", None)
    assert [[l.label for l in s] for s in shared] == [["1"], ["2 boxes", "2 labels"]]


def test_plain_svg_layers(tmp_path):
    """Top-level <g data-name> groups (Illustrator, VectorCraft) count as layers, as Inkscape layers do."""
    from llandau.media import svg_layers, _steps
    path = tmp_path / "plain.svg"
    path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><title>t</title>'
        '<g id="Layer_2" data-name="2"><rect width="1" height="1"/></g>'
        '<g id="Layer_1" data-name="1a"><g data-name="nested"><rect width="2" height="2"/></g></g>'
        '<g id="wrapper"><g id="Layer_3" data-name="_draft"><rect width="3" height="3"/></g></g>'
        '<rect width="4" height="4"/></svg>'
    )
    layers = svg_layers(path)
    assert [(l.id, l.label) for l in layers] == [("Layer_2", "2"), ("Layer_1", "1a"), ("Layer_3", "_draft")]
    assert [[l.id for l in s] for s in _steps(layers, "label", None)] == [["Layer_1"], ["Layer_2"]]


def test_vector_layers_drop_off_page(tmp_path):
    """Scratch objects outside the page are dropped in vector mode; those crossing the edge stay."""
    from llandau.media import layered_svg
    path = tmp_path / "offpage.svg"
    path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
        '<g id="L1" data-name="1"><rect id="inside" x="10" y="10" width="20" height="20"/>'
        '<rect id="edge" x="90" y="40" width="20" height="20"/>'
        '<rect id="outside" x="150" y="150" width="20" height="20"/></g></svg>'
    )
    steps = layered_svg(path, R.body, mode="vector")
    assert len(steps) == 1 and [len(layer.submobjects) for layer in steps[0]] == [2]


def test_fade_keeps_covering():
    """A fading group is blended as one layer: an opaque shape hides what lies beneath it within the group."""
    class Cover(Scene):
        def construct(self):
            self.camera.background_color = "#FFFFFF"
            dark = Square(color=BLACK, fill_opacity=1, stroke_width=0)
            light = Square(color="#FFFFFF", fill_opacity=1, stroke_width=0).scale(0.5)
            self.play(FadeIn(VGroup(dark, light)), run_time=1)
            self.play(FadeOut(VGroup(dark, light)), run_time=1)

    deck = build([Cover])
    for time in (0.25, 0.5, 0.75, 1.25, 1.5, 1.75):
        image = deck.render(time, 320, 180)
        assert image[90, 160, :3].min() > 250, f"the white square let the dark one through at t={time}"
    assert ink(deck.render(0.5, 320, 180)) > 0  # the dark square is on its way in


def test_fade_blends_with_prior_content():
    """A fade starts from what is already on the slide, not from the background."""
    class Over(Scene):
        def construct(self):
            self.camera.background_color = "#FFFFFF"
            self.add(Square(color=BLACK, fill_opacity=1, stroke_width=0))
            light = Square(color="#FFFFFF", fill_opacity=1, stroke_width=0).scale(0.5)
            self.play(FadeIn(light), run_time=1)
            self.play(FadeOut(light), run_time=1)

    deck = build([Over])
    center = lambda t: int(deck.render(t, 320, 180)[90, 160, :3].mean())  # noqa: E731
    assert center(0.0) < 10 and center(1.0) > 245 and center(2.0) < 10
    assert 80 < center(0.5) < 180 and 80 < center(1.5) < 180, (center(0.5), center(1.5))


def test_live_state_and_play_set_the_step():
    """A state dict applies at the step; a camera or a recording plays on the step's clock,
    which lasts long enough for it."""
    from llandau import live_scene
    from llandau.live import load_recording, load_state, CAMERA_MOVE

    @live_scene("atom")
    def atom(viewer):
        viewer.load(np.zeros((4, 1, 3), np.float32), np.array([6]))

    keyframes = [{"t": 0.0, "frame": 0, "fog": 0.0, "theme": "dark"}, {"t": 2.0, "frame": 3, "fog": 1.0}]
    assert load_recording(keyframes) == [(0.0, {"frame": 0, "fog": 0.0}), (2.0, {"frame": 3, "fog": 1.0})]
    assert load_state({"theme": "dark", "fog": 0.3}) == {"fog": 0.3}
    with pytest.raises(ValueError):
        load_recording([{"t": 1.0}, {"t": 0.5}])

    class Played(Deck):
        theme = LIGHT

        def construct(self):
            self.live("atom", R.body.left, still=False, play=keyframes)
            self.reveal(text("played"))
            self.live("atom", R.body.left, still=False, state={"fog": 0.5, "camera": {"azimuth": 10}})
            self.reveal(text("moved"))
            self.live("atom", R.body.left, still=False, frame=2)
            self.reveal(text("set"))

    deck = build([Played])
    marks = [t for t, _, _ in deck.marks]
    assert marks[0] >= 2.0  # the recording's 2 s, then the reveal
    assert marks[1] - marks[0] >= CAMERA_MOVE  # the move to the state's camera
    assert marks[2] - marks[1] < CAMERA_MOVE  # a frame is set at once
    assert deck.num_lives == 3
