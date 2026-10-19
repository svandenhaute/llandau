"""Live, interactive polyscope 3D on selected slide steps (manim-slides' Qt player, full screen).

    @live_scene("spiral")
    def spiral(ps):                       # gets the polyscope module; no import needed
        ps.register_point_cloud("spiral", points)

    class Results(Deck):
        def construct(self):
            self.reveal(text("before"))
            self.live("spiral")           # the next step shows the 3D window over the slide
            self.reveal(text("3D here"))  # ... until this step ends

    present render slides.py && present live slides.py

Full screen on the primary display; for a projector, let the OS mirror the screen.
Needs polyscope: pip install "present[live]".
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

__all__ = ["live_scene", "LIVE_PREFIX"]

LIVE_PREFIX = "live:"  # a step is live when its notes are LIVE_PREFIX + scene name
SCENES = {}


def live_scene(name: str):
    """Register a function that builds a polyscope scene (it is called with the polyscope module)."""
    def register(build):
        SCENES[name] = build
        return build
    return register


def run(names: list[str], folder: Path = Path("slides")) -> int:
    import polyscope as ps
    import polyscope.imgui as psim

    Path(".polyscope.ini").write_text(json.dumps({"windowPosX": 0, "windowPosY": 0}))
    ps.set_give_focus_on_show(False)
    ps.init()  # before Qt: GLFW overwrites the NSApp delegate

    from qtpy.QtCore import QTimer
    from manim_slides.config import Config, PresentationConfig
    from manim_slides.present.player import Player
    from manim_slides.qt_utils import qapp

    app = qapp()
    screen = app.primaryScreen().geometry()
    ps.set_window_size(screen.width(), screen.height())  # cover the slide while shown

    keys = {"next": False, "previous": False}

    def record_keys():
        # runs inside frame_tick() (ImGui is only valid there); acted on in tick()
        if psim.IsKeyPressed(psim.ImGuiKey_RightArrow, False) or psim.IsKeyPressed(psim.ImGuiKey_Space, False):
            keys["next"] = True
        if psim.IsKeyPressed(psim.ImGuiKey_LeftArrow, False):
            keys["previous"] = True

    ps.set_user_callback(record_keys)
    ps.set_open_imgui_window_for_user_callback(False)

    shown = {"scene": None}

    def show_scene(name: str):
        if name == shown["scene"]:
            return
        if name not in SCENES:
            raise KeyError(f"no @live_scene({name!r}) defined; known: {sorted(SCENES)}")
        ps.remove_all_structures()
        SCENES[name](ps)
        ps.reset_camera_to_home_view()
        shown["scene"] = name

    class LivePlayer(Player):
        def slide_changed_callback(self):
            super().slide_changed_callback()
            notes = self.current_slide_config.notes
            if notes.startswith(LIVE_PREFIX):
                show_scene(notes.removeprefix(LIVE_PREFIX).strip())
                timer.start()
            elif timer.isActive():
                timer.stop()
                ps.show(1)  # renders one frame, then hides the window
                self.activateWindow()
                self.raise_()

    def tick():
        ps.frame_tick()
        if keys["next"] or ps.window_requests_close():
            player.next()
        elif keys["previous"]:
            player.previous()
        keys["next"] = keys["previous"] = False

    timer = QTimer(interval=16)
    timer.timeout.connect(tick)
    player = LivePlayer(
        Config(),
        [PresentationConfig.from_file(folder / f"{name}.json") for name in names],
        full_screen=True,
        hide_info_window=True,
    )
    player.show(app.screens())

    exit_code = app.exec()
    timer.stop()
    ps.shutdown()
    return exit_code


def main(argv: list[str]) -> int:
    from .render import slide_names

    parser = argparse.ArgumentParser(prog="present live", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file", type=Path)
    parser.add_argument("--only", nargs="+", metavar="SLIDE", help="present only these slides")
    args = parser.parse_args(argv)
    names = args.only or slide_names(args.file)  # also runs the file, registering its @live_scene's
    return run(names)
