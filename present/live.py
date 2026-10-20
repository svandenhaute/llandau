"""Live, interactive 3D (molview) over chosen slide steps, in manim-slides' player.

    @live_scene("butane")
    def butane(viewer):                     # a molview.Viewer, in the deck's theme
        viewer.load(positions, numbers)     # (frames, atoms, 3), (atoms,)
        viewer.add_vectors("tangent", tangents, visible=False)

    class Results(Deck):
        def construct(self):
            self.live("butane", R.body.right)                  # the next step shows it over R.body.right
            self.reveal(place(text("dihedral scan"), R.body.left))
            self.live("butane", R.body.right, camera={"azimuth": 30}, frame=60, show=["tangent"])
            self.reveal(...)

    present render slides.py && present live slides.py

A view (the keywords of Deck.live) sets, when its step starts: camera (a dict as
viewer.camera; the camera moves there smoothly), frame, style, sidechains,
coloring, scrub_discrete, and show / hide (layer names). Consecutive steps of the
same scene keep everything else, including where the presenter turned it.

In the player: drag rotates, right drag pans, the wheel scrubs frames, pinch or
ctrl + wheel zooms, a double click resets the camera. Keys: , and . step frames,
P plays the trajectory, D toggles scrubbing the discrete states, 0 resets the
camera; the arrows still change slides. Rendered (video, HTML, PDF), a live step
shows a still image of its view instead.

Needs molview: pip install "present[live]" (and molview itself).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

__all__ = ["live_scene", "LIVE_PREFIX", "VIEW_KEYS"]

LIVE_PREFIX = "live:"  # a step is live when its notes are LIVE_PREFIX + json spec
SCENES = {}
VIEW_KEYS = {"camera", "frame", "style", "sidechains", "coloring", "scrub_discrete", "show", "hide"}
DEFAULT_REGION = (0.05, 0.15, 0.9, 0.8)  # left, top, width, height as fractions of the slide


def live_scene(name: str):
    """Register a function that builds a scene into the molview.Viewer it is given."""
    def register(build):
        SCENES[name] = build
        return build
    return register


def encode(scene: str, region: tuple, theme: str, view: dict) -> str:
    unknown = set(view) - VIEW_KEYS
    if unknown:
        raise TypeError(f"unknown view settings {sorted(unknown)}; known: {sorted(VIEW_KEYS)}")
    return LIVE_PREFIX + json.dumps({"scene": scene, "region": list(region), "theme": theme, "view": view})


def decode(notes: str) -> dict | None:
    """The live spec of a step's notes, or None."""
    if not notes.startswith(LIVE_PREFIX):
        return None
    body = notes.removeprefix(LIVE_PREFIX).strip()
    if not body.startswith("{"):  # "live:name"
        return {"scene": body, "region": list(DEFAULT_REGION), "theme": "dark", "view": {}}
    return json.loads(body)


def build(viewer, scene: str):
    if scene not in SCENES:
        raise KeyError(f"no @live_scene({scene!r}) defined; known: {sorted(SCENES)}")
    viewer.reset()
    SCENES[scene](viewer)


def apply_view(viewer, view: dict, camera: bool = True):
    """Apply a view's settings; the camera only with `camera` (else it is animated by the caller)."""
    for key in ("style", "sidechains"):
        if key in view:
            setattr(viewer, key, view[key])
    if "coloring" in view:
        viewer.coloring = view["coloring"]
    if "scrub_discrete" in view:
        viewer.scrub_discrete = view["scrub_discrete"]
    for layer in view.get("show", []):
        viewer.set_visible(layer, True)
    for layer in view.get("hide", []):
        viewer.set_visible(layer, False)
    if "frame" in view:
        viewer.frame = view["frame"]
    if camera and "camera" in view:
        viewer.camera = view["camera"]


def still(scene: str, theme: str, view: dict, size: tuple[int, int]):
    """RGBA image (height, width, 4) of a scene's view, for the rendered slides."""
    import molview

    viewer = molview.Viewer(theme=theme)
    build(viewer, scene)
    apply_view(viewer, view)
    return viewer.render_image(*size)


def blend_camera(start: dict, end: dict, t: float) -> dict:
    """Camera between two camera dicts; angles take the short way around."""
    camera = dict(end)
    for key in ("elevation", "fov"):
        if key in start and key in end:
            camera[key] = start[key] + t * (end[key] - start[key])
    if "azimuth" in start and "azimuth" in end:
        turn = (end["azimuth"] - start["azimuth"] + 180.0) % 360.0 - 180.0
        camera["azimuth"] = start["azimuth"] + t * turn
    if "distance" in start and "distance" in end:
        camera["distance"] = start["distance"] * (end["distance"] / start["distance"]) ** t
    if "target" in start and "target" in end:
        camera["target"] = [a + t * (b - a) for a, b in zip(start["target"], end["target"])]
    return camera


def run(names: list[str], folder: Path = Path("slides"), full_screen: bool = True) -> int:
    import molview
    from manim_slides.config import Config, PresentationConfig
    from manim_slides.present.player import Player
    from manim_slides.qt_utils import qapp
    from molview.qt import ViewerWidget
    from qtpy.QtCore import QElapsedTimer, QPoint, QRect, Qt, QTimer

    app = qapp()

    class LivePlayer(Player):
        def __init__(self, *args, **kwargs):
            self.live = None  # Player.__init__ already changes slides
            super().__init__(*args, **kwargs)
            # an overlay window: as a child widget, the slide video no longer shows on macOS
            self.live = ViewerWidget(self, viewer=molview.Viewer(), overlay=True)
            self.spec = None
            self.playing = QTimer(self, interval=33)
            self.playing.timeout.connect(lambda: self.step_frame(1, wrap=True))
            self.moving = QTimer(self, interval=16)  # camera animation
            self.moving.timeout.connect(self.move_camera)
            self.clock = QElapsedTimer()
            self.camera_path = None
            self.slide_changed_callback()

        def slide_area(self) -> QRect:
            """Where the slide is drawn in the window (the video keeps its aspect ratio)."""
            width, height = self.current_presentation_config.resolution
            area = self.centralWidget().geometry()
            scale = min(area.width() / width, area.height() / height)
            w, h = round(width * scale), round(height * scale)
            return QRect(area.x() + (area.width() - w) // 2, area.y() + (area.height() - h) // 2, w, h)

        def place_live(self):
            if self.spec is None:
                return
            slide = self.slide_area()
            left, top, width, height = self.spec["region"]
            corner = QPoint(round(slide.x() + left * slide.width()), round(slide.y() + top * slide.height()))
            corner = self.mapToGlobal(corner)
            self.live.setGeometry(corner.x(), corner.y(), round(width * slide.width()), round(height * slide.height()))

        def resizeEvent(self, event):  # noqa: N802
            super().resizeEvent(event)
            self.place_live()

        def moveEvent(self, event):  # noqa: N802
            super().moveEvent(event)
            self.place_live()

        def slide_changed_callback(self):
            super().slide_changed_callback()
            if self.live is None:
                return
            spec = decode(self.current_slide_config.notes)
            viewer = self.live.viewer
            self.playing.stop()
            if spec is None:
                self.spec = None
                self.moving.stop()
                self.live.hide()
                return
            same_scene = self.spec is not None and self.spec["scene"] == spec["scene"]
            self.spec = spec
            if not same_scene:
                viewer.theme = spec["theme"]
                build(viewer, spec["scene"])
            apply_view(viewer, spec["view"], camera=not same_scene)
            if same_scene and "camera" in spec["view"]:
                self.camera_path = (viewer.camera, spec["view"]["camera"])
                self.clock.start()
                self.moving.start()
            self.place_live()
            self.live.show()
            self.live.raise_()
            self.live.request_render()

        def move_camera(self):
            start, end = self.camera_path
            t = min(self.clock.elapsed() / 600.0, 1.0)
            eased = 0.5 - 0.5 * math.cos(math.pi * t)
            self.live.viewer.camera = blend_camera(start, end, eased)
            self.live.request_render()
            if t >= 1.0:
                self.moving.stop()

        def step_frame(self, steps: int, wrap: bool = False):
            viewer = self.live.viewer
            count = max(viewer.num_frames, 1)
            frame = viewer.frame + steps
            viewer.frame = frame % count if wrap else max(0, min(frame, count - 1))
            self.live.request_render()

        def keyPressEvent(self, event):  # noqa: N802
            if self.spec is not None:
                key, viewer = event.key(), self.live.viewer
                if key in (Qt.Key.Key_Comma, Qt.Key.Key_Period):
                    self.step_frame(1 if key == Qt.Key.Key_Period else -1)
                    return
                if key == Qt.Key.Key_P:
                    self.playing.stop() if self.playing.isActive() else self.playing.start()
                    return
                if key == Qt.Key.Key_D and viewer.num_discrete_frames:
                    viewer.scrub_discrete = not viewer.scrub_discrete
                    self.live.request_render()
                    return
                if key == Qt.Key.Key_0:
                    viewer.reset_camera()
                    self.live.request_render()
                    return
            super().keyPressEvent(event)

    player = LivePlayer(
        Config(),
        [PresentationConfig.from_file(folder / f"{name}.json") for name in names],
        full_screen=full_screen,
        hide_info_window=True,
    )
    player.show(app.screens())
    return app.exec()


def main(argv: list[str]) -> int:
    from .render import slide_names

    parser = argparse.ArgumentParser(prog="present live", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file", type=Path)
    parser.add_argument("--only", nargs="+", metavar="SLIDE", help="present only these slides")
    parser.add_argument("--window", action="store_true", help="not full screen")
    args = parser.parse_args(argv)
    names = args.only or slide_names(args.file)  # also runs the file, registering its @live_scene's
    return run(names, full_screen=not args.window)
