"""Live, interactive 3D (llandau.Viewer) over chosen slide steps.

    @live_scene("protein")
    def protein(viewer):                    # a llandau.Viewer, in the deck's theme
        viewer.load(positions, numbers)     # (frames, atoms, 3), (atoms,)
        viewer.add_vectors("tangent", tangents, visible=False)
        viewer.set_plots({"energy": energy})  # under the view on G, or with plots=True

    class Results(Deck):
        def construct(self):
            self.live("protein", R.body.right)                 # the next step shows it over R.body.right
            self.reveal(place(text("the NMR models"), R.body.left))
            self.live("protein", R.body.right, camera={"azimuth": 30}, frame=20, show=["tangent"])
            self.reveal(...)

    llandau play slides.py

A view (the keywords of Deck.live) sets, when its step starts: camera (a dict as
viewer.camera; the camera moves there smoothly), frame, style, sidechains,
shading, fog, occlusion, coloring, scrub_discrete, show / hide (layer names), and
plots (True: the viewer's plots large under its 3D view).
Consecutive steps of the same scene keep everything else, including where the
presenter turned it.

In the player: drag rotates, right drag pans, the wheel scrubs frames, pinch or
ctrl + wheel zooms, a double click resets the camera; ? over the view shows
every key (the explorer's, with the arrows still changing steps). Exported
(llandau pdf), a live step shows a still image of its 3D view (without plots).

Several views on one step: call live() once per region. With opens=R.body a
click on a view opens it up to the body over the others (a grid of molecules,
each opening for a closer look) and the next click closes it:

    for name, cell in zip(names, R.body.grid(2, 5, gap=0.2)):
        self.live(name, cell, opens=R.body)
"""
from __future__ import annotations

__all__ = ["live_scene", "VIEW_KEYS"]

SCENES = {}
# view key -> viewer attribute, in the order applied (frame after scrub_discrete)
_ATTRIBUTES = {"style": "style", "sidechains": "sidechains", "shading": "shading", "fog": "fog", "occlusion": "occlusion",
               "coloring": "coloring", "scrub_discrete": "scrub_discrete", "frame": "frame", "plots": "show_plots"}
VIEW_KEYS = set(_ATTRIBUTES) | {"show", "hide", "camera"}


def live_scene(name: str):
    """Register a function that builds a scene into the llandau.Viewer it is given."""
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
    for key, attribute in _ATTRIBUTES.items():
        if key in view:
            setattr(viewer, attribute, bool(view[key]) if key == "plots" else view[key])
    for layer in view.get("show", []):
        viewer.set_visible(layer, True)
    for layer in view.get("hide", []):
        viewer.set_visible(layer, False)
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
