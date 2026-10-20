"""A Qt widget showing a viewer.

    widget = ViewerWidget(parent)
    widget.viewer.load(positions, numbers)

Two ways to get the pictures on screen:
  native  wgpu draws straight into the widget's own native window (macOS, Windows;
          needed for overlays over the manim-slides player)
  raster  the viewer renders offscreen and the widget paints the image, like any
          Qt widget (works everywhere, e.g. Wayland; a copy per frame)
The default is native on macOS and Windows and raster elsewhere; the environment
variable MOLVIEW_QT=native|raster or the `mode` argument overrides it.

Mouse: left drag rotates, right drag (or shift + left) pans, wheel scrubs the
trajectory, ctrl + wheel or a pinch zooms, double click resets the camera.
Keyboard input is left to the parent (e.g. the manim-slides player).
"""
from __future__ import annotations

import os
import sys

from qtpy.QtCore import QEvent, QPointF, Qt, QTimer, Signal
from qtpy.QtGui import QImage, QPainter
from qtpy.QtWidgets import QWidget

from . import Viewer

__all__ = ["ViewerWidget", "default_mode"]


def default_mode() -> str:
    mode = os.environ.get("MOLVIEW_QT")
    if mode in ("native", "raster"):
        return mode
    return "native" if sys.platform in ("darwin", "win32") else "raster"


class ViewerWidget(QWidget):
    """`overlay=True` makes it a frameless window floating over `parent` (which
    must then place it in global coordinates) instead of a child widget."""

    frame_changed = Signal(int)  # when the wheel scrubs

    def __init__(
        self,
        parent: QWidget | None = None,
        viewer: Viewer | None = None,
        overlay: bool = False,
        mode: str | None = None,
    ):
        flags = Qt.WindowType.Widget
        if overlay:
            flags = (
                Qt.WindowType.Tool
                | Qt.WindowType.FramelessWindowHint
                | Qt.WindowType.WindowDoesNotAcceptFocus
            )
        super().__init__(parent, flags)
        self.viewer = viewer or Viewer()
        self.mode = mode or default_mode()
        if overlay:
            self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        if self.mode == "native":
            # our own native window, drawn by wgpu instead of Qt; the parents stay as they are
            self.setAttribute(Qt.WidgetAttribute.WA_DontCreateNativeAncestors)
            self.setAttribute(Qt.WidgetAttribute.WA_NativeWindow)
            self.setAttribute(Qt.WidgetAttribute.WA_PaintOnScreen)
            self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._attached = False
        self._last: QPointF | None = None
        self._scroll = 0.0  # accumulated wheel delta, in frames
        # Render at most once per event loop pass, however many inputs arrive.
        self._timer = QTimer(self, singleShot=True, interval=0)
        self._timer.timeout.connect(self._render)

    def paintEngine(self):  # noqa: N802
        return None if self.mode == "native" else super().paintEngine()  # native: wgpu draws, not Qt

    def _pixels(self) -> tuple[int, int]:
        ratio = self.devicePixelRatioF()
        return max(1, round(self.width() * ratio)), max(1, round(self.height() * ratio))

    def request_render(self):
        if self.mode == "raster":
            self.update()  # Qt coalesces these into one paint
        elif not self._timer.isActive():
            self._timer.start()

    def _render(self):
        if self._attached and self.isVisible():
            self.viewer.render()

    # --- window lifecycle --------------------------------------------------------

    def showEvent(self, event):  # noqa: N802
        super().showEvent(event)
        if self.mode == "native" and not self._attached:
            self.viewer.attach(int(self.winId()), *self._pixels())
            self._attached = True
        self.request_render()

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        if self._attached:
            self.viewer.resize(*self._pixels())
        self.request_render()

    def paintEvent(self, event):  # noqa: N802
        if self.mode == "native":
            self.request_render()
            return
        width, height = self._pixels()
        pixels = self.viewer.render_image(width, height)
        image = QImage(pixels.data, width, height, 4 * width, QImage.Format.Format_RGBA8888)
        image.setDevicePixelRatio(self.devicePixelRatioF())
        painter = QPainter(self)
        painter.drawImage(0, 0, image)
        painter.end()

    def event(self, event):
        if event.type() == QEvent.Type.NativeGesture and event.gestureType() == Qt.NativeGestureType.ZoomNativeGesture:
            self.viewer.zoom(1.0 / (1.0 + event.value()))
            self.request_render()
            return True
        return super().event(event)

    def closeEvent(self, event):  # noqa: N802
        if self._attached:
            self.viewer.detach()
            self._attached = False
        super().closeEvent(event)

    # --- mouse -----------------------------------------------------------------

    def mousePressEvent(self, event):  # noqa: N802
        self._last = event.position()

    def mouseReleaseEvent(self, event):  # noqa: N802
        self._last = None

    def mouseDoubleClickEvent(self, event):  # noqa: N802
        self.viewer.reset_camera()
        self.request_render()

    def mouseMoveEvent(self, event):  # noqa: N802
        if self._last is None:
            return
        position = event.position()
        ratio = self.devicePixelRatioF()
        dx, dy = (position.x() - self._last.x()) * ratio, (position.y() - self._last.y()) * ratio
        self._last = position
        pan = event.buttons() & Qt.MouseButton.RightButton or (
            event.modifiers() & Qt.KeyboardModifier.ShiftModifier
        )
        if pan:
            self.viewer.pan(dx, dy, self._pixels()[1])
        else:
            self.viewer.orbit(dx, dy)
        self.request_render()

    def wheelEvent(self, event):  # noqa: N802
        # trackpads report pixels, mice report angles (120 per notch)
        pixels, angle = event.pixelDelta(), event.angleDelta()
        delta = pixels.y() / 8.0 if not pixels.isNull() else angle.y() / 120.0
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.viewer.zoom(0.9 ** delta)
        else:
            self._scroll -= delta
            steps = int(self._scroll)
            if steps:
                self._scroll -= steps
                self.viewer.frame = max(0, self.viewer.frame + steps)
                self.frame_changed.emit(self.viewer.frame)
        self.request_render()
