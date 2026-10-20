"""Offscreen rendering of every layer, and the Qt widget in raster mode.

CI runs these with software rendering on Linux (Mesa's lavapipe)."""
import os

import numpy as np
import pytest

from riemann_viewer import Viewer

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


def test_ghost_and_discrete(viewer):
    viewer.set_ghost(POSITIONS[-1] + 1.0)
    viewer.set_discrete_frames(POSITIONS[::2])
    viewer.scrub_discrete = True
    assert viewer.num_frames == 3
    assert drawn(viewer.render_image(160, 120)) > 0.01


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


def test_qt_raster_widget(viewer):
    pytest.importorskip("qtpy")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qtpy.QtWidgets import QApplication

    from riemann_viewer.qt import ViewerWidget

    app = QApplication.instance() or QApplication([])
    widget = ViewerWidget(viewer=viewer, mode="raster")
    widget.resize(200, 150)
    widget.show()
    app.processEvents()
    image = widget.grab().toImage()
    pixels = np.frombuffer(image.constBits(), np.uint8, image.sizeInBytes()).reshape(image.height(), -1, 4)
    assert drawn(pixels) > 0.01
    widget.close()


def test_explorer_window(viewer, tmp_path):
    """The native explorer opens, draws one frame (3D view and panel) and closes."""
    import sys

    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        pytest.skip("no display")
    from riemann_viewer.explorer import explore

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
