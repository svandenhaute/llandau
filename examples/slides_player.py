"""The viewer inside manim-slides' player, over the slide, on steps whose notes start with "live:".

    python slides_player.py ~/sandbox/polyscope-slides/slides/Dummy.json

The viewer is a frameless window floating over the player (as a native child
widget, the player's video no longer shows on macOS).

The 3D region is given as a fraction of the slide (left, top, width, height),
so it lines up with the slide however the player is letterboxed.
"""
import sys
from pathlib import Path

from manim_slides.config import Config, PresentationConfig
from manim_slides.present.player import Player
from manim_slides.qt_utils import qapp
from qtpy.QtCore import QPoint, QRect

from butane import butane
from riemann_viewer.qt import ViewerWidget

REGION = (0.08, 0.2, 0.84, 0.75)  # left, top, width, height as fractions of the slide
THEME = "dark"  # as the slides (present's DARK theme)


class LivePlayer(Player):
    def __init__(self, *args, **kwargs):
        self.live = None  # Player.__init__ already changes slides
        super().__init__(*args, **kwargs)
        self.live = ViewerWidget(self, overlay=True)
        self.live.viewer.theme = THEME
        self.live.viewer.load(*butane())
        self.live.viewer.set_sizes(radius_scale=0.6)
        self.slide_changed_callback()

    def slide_area(self) -> QRect:
        """Where the slide is drawn inside the window (the video keeps its aspect ratio)."""
        width, height = self.current_presentation_config.resolution
        area = self.centralWidget().geometry()
        scale = min(area.width() / width, area.height() / height)
        w, h = round(width * scale), round(height * scale)
        return QRect(area.x() + (area.width() - w) // 2, area.y() + (area.height() - h) // 2, w, h)

    def place_live(self):
        slide = self.slide_area()
        left, top, width, height = REGION
        corner = self.mapToGlobal(QPoint(round(slide.x() + left * slide.width()), round(slide.y() + top * slide.height())))
        self.live.setGeometry(corner.x(), corner.y(), round(width * slide.width()), round(height * slide.height()))

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        if self.live:
            self.place_live()

    def moveEvent(self, event):  # noqa: N802
        super().moveEvent(event)
        if self.live:
            self.place_live()

    def slide_changed_callback(self):
        super().slide_changed_callback()
        if self.live is None:
            return
        if self.current_slide_config.notes.startswith("live:"):
            self.place_live()
            self.live.show()
            self.live.raise_()
        else:
            self.live.hide()


def main():
    path = Path(sys.argv[1]).expanduser()
    app = qapp()
    player = LivePlayer(Config(), [PresentationConfig.from_file(path)], hide_info_window=True)
    player.show(app.screens())
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
