"""Offscreen rendering of every layer, and the Qt widget in raster mode.

CI runs these with software rendering on Linux (Mesa's lavapipe)."""
import os

import numpy as np
import pytest

from llandau import Viewer

# water, stretching one O-H bond over 5 frames
START = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
STRETCH = np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.0, 0.0, 0.0]])
POSITIONS = START + np.linspace(0, 1, 5)[:, None, None] * STRETCH
NUMBERS = np.array([8, 1, 1])


def drawn(image: np.ndarray, background=255) -> float:
    """Fraction of pixels that differ from the background."""
    return float((np.abs(image[..., :3].astype(int) - background).max(axis=-1) > 8).mean())


@pytest.fixture
def viewer():
    return Viewer().load(POSITIONS, NUMBERS)


def test_adapter(viewer):
    print(viewer.adapter)
    assert viewer.adapter


def test_atoms(viewer):
    image = viewer.render_image(160, 120)
    assert image.shape == (120, 160, 4) and image.dtype == np.uint8
    assert 0.01 < drawn(image) < 0.9


def test_frames_move_atoms(viewer):
    first = viewer.render_image(160, 120)
    viewer.frame = 4
    assert viewer.frame == 4
    assert (viewer.render_image(160, 120) != first).any()


def test_layers(viewer):
    viewer.set_visible("atoms", False)
    viewer.set_visible("trails", False)
    assert drawn(viewer.render_image(160, 120)) == 0
    viewer.add_vectors("tangent", np.gradient(POSITIONS, axis=0), scale=3.0)
    assert drawn(viewer.render_image(160, 120)) > 0
    viewer.set_visible("tangent", False)
    viewer.set_visible("surface", True)
    viewer.set_surface(opacity=1.0)
    assert drawn(viewer.render_image(160, 120)) > 0.05


def test_trail_window(viewer):
    """Only the segments within the window of the current frame are drawn: the
    segments that should be inside are red, the rest blue."""
    viewer.set_visible("atoms", False)
    viewer.set_sizes(trail_radius=0.1)
    viewer.fog = 0.0

    def colors_drawn(frame, window, inside):
        colors = np.tile([0.0, 0.0, 1.0, 1.0], (len(POSITIONS) - 1, 1))
        colors[inside] = [1.0, 0.0, 0.0, 1.0]
        viewer.set_trails(colors=colors)
        viewer.frame, viewer.trail_window = frame, window
        image = viewer.render_image(160, 120).astype(int)
        red = (image[..., 0] > image[..., 2] + 40).any()
        blue = (image[..., 2] > image[..., 0] + 40).any()
        return red, blue

    assert viewer.trail_window is None
    assert colors_drawn(2, None, slice(0, 4)) == (True, False)
    assert colors_drawn(2, 1, slice(1, 3)) == (True, False)  # frames 1 to 3
    assert colors_drawn(0, 1, slice(0, 1)) == (True, False)  # clamped at the ends
    assert colors_drawn(4, 1, slice(3, 4)) == (True, False)
    assert colors_drawn(2, 0, slice(0, 0)) == (False, False)


def test_ghost_and_discrete(viewer):
    viewer.set_ghost(POSITIONS[-1] + 1.0)
    viewer.set_discrete_frames(POSITIONS[::2])
    viewer.scrub_discrete = True
    assert viewer.num_frames == 3
    assert drawn(viewer.render_image(160, 120)) > 0.01


def test_discrete_states_at_their_own_path_frames(viewer):
    """Discrete states may say which path frame they are at (fractional), for the trails
    and the plots' cursor; the list must match the states and stay within the path."""
    viewer.set_discrete_frames(POSITIONS[::2], path_frames=[0, 0.5, 4])
    viewer.scrub_discrete = True
    assert viewer.num_frames == 3
    assert drawn(viewer.render_image(160, 120)) > 0.01
    with pytest.raises(ValueError):
        viewer.set_discrete_frames(POSITIONS[::2], path_frames=[0, 1])
    with pytest.raises(ValueError):
        viewer.set_discrete_frames(POSITIONS[::2], path_frames=[0, 1, 4.5])
    viewer.set_discrete_frames(None)
    assert not viewer.scrub_discrete


def test_discrete_trails(viewer):
    """Thick trails through the discrete states, a layer of their own, off until asked for."""
    states = POSITIONS[::2] + np.arange(3)[:, None, None] * 1.0  # three states, 1 A apart
    viewer.set_discrete_frames(states)
    assert viewer.state["layers"]["discrete trails"] is False
    plain = drawn(viewer.render_image(160, 120))
    viewer.set_discrete_trails(radius=0.1)
    assert viewer.state["layers"]["discrete trails"] is True
    assert drawn(viewer.render_image(160, 120)) > plain + 0.005
    viewer.set_visible("discrete trails", False)
    assert drawn(viewer.render_image(160, 120)) == pytest.approx(plain, abs=0.002)
    with pytest.raises(ValueError):
        viewer.set_discrete_trails(color=[1.0, 0.0])


def test_flat_shading(viewer):
    lit = viewer.render_image(160, 120)
    viewer.shading = "flat"
    assert viewer.shading == "flat"
    flat = viewer.render_image(160, 120)
    assert drawn(flat) > 0.01
    assert np.abs(flat.astype(int) - lit.astype(int)).max() > 30
    with pytest.raises(ValueError):
        viewer.shading = "glossy"


def test_dark_theme(viewer):
    viewer.theme = "dark"
    image = viewer.render_image(160, 120)
    assert image[0, 0, :3].max() < 10  # black background
    assert drawn(image, background=0) > 0.01


def test_explorer_window(viewer, tmp_path):
    """The native explorer opens, draws one frame (3D view and panel) and closes."""
    import sys

    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        pytest.skip("no display")
    from llandau.explorer import explore

    path = tmp_path / "explorer.png"
    explore(viewer, plots={"x": np.arange(5.0)}, info="test",
            buttons={"target": lambda camera: str(camera["target"])}, screenshot=str(path))
    from PIL import Image

    image = np.asarray(Image.open(path).convert("RGB"))
    assert image.shape[0] > 100 and image.shape[1] > 300
    panel, view = image[:, -50:], image[:, : image.shape[1] // 2]
    assert drawn(view) > 0.001  # the molecule
    assert (panel != 255).any()  # the panel
    explore(viewer, screenshot=str(path))  # a second window in the same process


def test_slides_timeline():
    from llandau._core import Slides, parse_svg

    svg = parse_svg(b'<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">'
                    b'<rect x="2" y="2" width="6" height="6" fill="#000"/></svg>')
    points = svg["paths"][0]["points"]
    assert points.shape == (16, 2)
    slides = Slides(frame=(16.0, 9.0), background=(1, 1, 1, 1))
    square = slides.add_shape((points - 5).astype(np.float32), np.array([0, 0, 0, 1, 0, 0, 0, 0, 0], np.float32))
    slides.add_clip("write", square, square, 0.0, 0.0, float("inf"), (0.0, 1.0))
    slides.add_mark(1.0)
    assert slides.duration == 1.0
    start, middle, end = (slides.render(t, 160, 90) for t in (0.0, 0.75, 1.0))
    ink = lambda image: 1 - image[..., :3].mean() / 255  # noqa: E731
    assert ink(start) == 0
    # halfway through filling in: half as dark
    assert ink(middle) == pytest.approx(ink(end) / 2, rel=0.1)
    assert ink(end) == pytest.approx(36 / 144, abs=0.02)  # 6 x 6 of the 16 x 9 frame


def test_plots_belong_to_the_viewer(viewer):
    from llandau.live import apply_view, check
    viewer.set_plots({"energy": np.arange(5.0)})  # values per path frame
    viewer.set_plots({"energy": (np.linspace(0, 1, 9), np.zeros(9))}, t=np.linspace(0, 1, 5))
    with pytest.raises(ValueError):
        viewer.set_plots({"energy": (np.arange(3.0), np.arange(4.0))})
    assert not viewer.show_plots
    check({"plots": True})
    apply_view(viewer, {"plots": True})
    assert viewer.show_plots
    viewer.reset()
    assert not viewer.show_plots


def test_fog_is_off_for_small_structures_unless_set():
    from llandau import Viewer

    small = Viewer().load(np.array([[[0.0, 0, 0], [1.5, 0, 0]]]), np.array([6, 6]))
    assert small.fog == 0.0
    big = Viewer().load(np.array([[[-9.0, 0, 0], [9.0, 0, 0]]]), np.array([6, 6]))
    assert big.fog > 0.3
    big.fog = 0.2
    big.load(np.array([[[0.0, 0, 0], [1.5, 0, 0]]]), np.array([6, 6]))
    assert abs(big.fog - 0.2) < 1e-6  # set by hand: kept
