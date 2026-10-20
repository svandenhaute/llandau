import numpy as np
import pytest

from present import (
    BLACK, BLUE, Create, Deck, Dot, FadeIn, FadeOut, LaggedStart, LIGHT, R, Rectangle, Scene, Square, Transform,
    VGroup, Write, build, place, text,
)
from present.mobject import smooth_handles


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


def test_spline_handles_are_manims():
    anchors = np.array([[0, 0, 0], [1, 2, 0], [3, 1, 0], [4, 4, 0]], float)
    h1, h2 = smooth_handles(anchors)
    # C1 and C2 continuity at the inner anchors
    for i in (1, 2):
        assert np.allclose(h2[i - 1] + h1[i], 2 * anchors[i])
        assert np.allclose(h1[i - 1] - 2 * h2[i - 1], h2[i] - 2 * h1[i])
