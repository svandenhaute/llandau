"""Live, interactive 3D (landau.Viewer) over chosen slide steps.

    @live_scene("butane")
    def butane(viewer):                     # a landau.Viewer, in the deck's theme
        viewer.load(positions, numbers)     # (frames, atoms, 3), (atoms,)
        viewer.add_vectors("tangent", tangents, visible=False)

    class Results(Deck):
        def construct(self):
            self.live("butane", R.body.right)                  # the next step shows it over R.body.right
            self.reveal(place(text("dihedral scan"), R.body.left))
            self.live("butane", R.body.right, camera={"azimuth": 30}, frame=60, show=["tangent"])
            self.reveal(...)

    landau play slides.py

A view (the keywords of Deck.live) sets, when its step starts: camera (a dict as
viewer.camera; the camera moves there smoothly), frame, style, sidechains,
shading, fog, coloring, scrub_discrete, and show / hide (layer names). Consecutive
steps of the same scene keep everything else, including where the presenter
turned it.

In the player: drag rotates, right drag pans, the wheel scrubs frames, pinch or
ctrl + wheel zooms, a double click resets the camera. Keys: , and . step frames,
P plays the trajectory, D toggles scrubbing the discrete states, 0 resets the
camera; the arrows still change steps. Exported (landau pdf), a live step shows
a still image of its view.
"""
from __future__ import annotations

__all__ = ["live_scene", "VIEW_KEYS"]

SCENES = {}
VIEW_KEYS = {"camera", "frame", "style", "sidechains", "shading", "fog", "coloring", "scrub_discrete", "show", "hide"}


def live_scene(name: str):
    """Register a function that builds a scene into the landau.Viewer it is given."""
    def register(build):
        SCENES[name] = build
        return build
    return register


def build(viewer, scene: str):
    if scene not in SCENES:
        raise KeyError(f"no @live_scene({scene!r}) defined; known: {sorted(SCENES)}")
    viewer.reset()
    SCENES[scene](viewer)


def apply_view(viewer, view: dict, camera: bool = True):
    """Apply a view's settings; the camera only with `camera` (else it is animated by the caller)."""
    for key in ("style", "sidechains", "shading", "fog"):
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
    from .viewer import Viewer

    viewer = Viewer(theme=theme)
    build(viewer, scene)
    apply_view(viewer, view)
    return viewer.render_image(*size)


def check(view: dict):
    unknown = set(view) - VIEW_KEYS
    if unknown:
        raise TypeError(f"unknown view settings {sorted(unknown)}; known: {sorted(VIEW_KEYS)}")
