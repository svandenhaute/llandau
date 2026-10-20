"""A window to explore a trajectory: the viewer plus a panel of controls and plots.

    from riemann_viewer import Viewer
    from riemann_viewer.explorer import explore

    viewer = Viewer().load(positions, numbers, roles=roles)
    viewer.add_vectors("tangent", tangents)
    explore(viewer, plots={"energy": energy})   # blocks until the window closes
"""
from __future__ import annotations

import json

import numpy as np
from qtpy.QtCore import QPointF, Qt, QTimer
from qtpy.QtGui import QColor, QPainter, QPainterPath, QPen
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .qt import ViewerWidget

__all__ = ["Explorer", "explore"]


class Plot(QWidget):
    """A scalar curve over the frames, with a marker at the current frame."""

    def __init__(self, name: str, xs: np.ndarray, ys: np.ndarray):
        super().__init__()
        self.name, self.xs, self.ys = name, np.asarray(xs, float), np.asarray(ys, float)
        finite = np.isfinite(self.ys)
        self.lo, self.hi = (self.ys[finite].min(), self.ys[finite].max()) if finite.any() else (0.0, 1.0)
        self.x = float(self.xs[0])
        self.setMinimumHeight(70)

    def set_x(self, x: float):
        self.x = x
        self.update()

    def value(self) -> float:
        return float(np.interp(self.x, self.xs, self.ys))

    def paintEvent(self, event):  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        text_height = painter.fontMetrics().height()
        painter.drawText(0, text_height - 3, f"{self.name}: {self.value():.4g}")
        left, top, width, height = 4.0, text_height + 2.0, self.width() - 8.0, self.height() - text_height - 6.0
        x0, x1 = self.xs.min(), self.xs.max()
        span, rise = (x1 - x0) or 1.0, (self.hi - self.lo) or 1.0

        def point(x, y):
            y = np.clip(np.nan_to_num(y, nan=self.lo), self.lo, self.hi)
            return QPointF(left + (x - x0) / span * width, top + (1 - (y - self.lo) / rise) * height)

        painter.setPen(QPen(QColor(200, 200, 200), 1))
        painter.drawRect(int(left), int(top), int(width), int(height))
        path = QPainterPath(point(self.xs[0], self.ys[0]))
        for x, y in zip(self.xs[1:], self.ys[1:]):
            path.lineTo(point(x, y))
        painter.setPen(QPen(QColor(40, 40, 40), 1.5))
        painter.drawPath(path)
        cursor = point(self.x, self.value())
        painter.setPen(QPen(QColor(150, 150, 150), 1))
        painter.drawLine(QPointF(cursor.x(), top), QPointF(cursor.x(), top + height))
        painter.setBrush(QColor(220, 50, 50))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(cursor, 3.5, 3.5)


class Explorer(QWidget):
    def __init__(
        self,
        viewer,
        plots: dict | None = None,
        t: np.ndarray | None = None,
        info: str | None = None,
        buttons: dict | None = None,
        title: str = "riemann_viewer",
    ):
        """`plots`: name -> values per frame, or name -> (xs, values) on the scale of `t`
        (the parameter of each frame; default the frame index). `info`: a line of
        text. `buttons`: label -> function of the viewer returning text to copy."""
        super().__init__()
        self.setWindowTitle(title)
        self.view = ViewerWidget(viewer=viewer)
        self.view.setMinimumSize(500, 400)
        self.viewer = viewer
        frames = max(viewer.num_path_frames, 1)
        self.t = np.arange(frames, dtype=float) if t is None else np.asarray(t, float)

        panel = QWidget()
        panel.setMinimumWidth(300)
        layout = QVBoxLayout(panel)
        if info:
            layout.addWidget(QLabel(info))
        form = QFormLayout()
        layout.addLayout(form)

        # frames
        self.slider = QSlider(Qt.Orientation.Horizontal, minimum=0, maximum=frames - 1)
        self.slider.valueChanged.connect(self.set_frame)
        self.view.frame_changed.connect(self.slider.setValue)
        self.play = QPushButton("Play", checkable=True)
        self.play.toggled.connect(lambda on: self.timer.start() if on else self.timer.stop())
        self.timer = QTimer(self, interval=33)
        self.timer.timeout.connect(lambda: self.slider.setValue((self.slider.value() + 1) % (self.slider.maximum() + 1)))
        self.frame_label = QLabel()
        row = QHBoxLayout()
        row.addWidget(self.play)
        row.addWidget(self.frame_label)
        if viewer.num_discrete_frames:
            # walk the discrete states instead of the path frames
            discrete = QCheckBox("scrub discrete", checked=viewer.scrub_discrete)
            discrete.toggled.connect(self.set_scrub_discrete)
            row.addWidget(discrete)
        form.addRow("frame", self.slider)
        form.addRow("", row)

        # appearance
        def combo(options, current, apply):
            box = QComboBox()
            box.addItems(options)
            box.setCurrentText(current)
            box.currentTextChanged.connect(lambda text: (apply(text), self.view.request_render()))
            return box

        def setter(name):
            return lambda value: setattr(viewer, name, value)

        form.addRow("theme", combo(["light", "dark"], viewer.theme, setter("theme")))
        styles = ["atoms", "protein"] if viewer.has_roles else ["atoms"]
        form.addRow("style", combo(styles, viewer.style, setter("style")))
        if "protein" in styles:
            form.addRow("side chains", combo(["sticks", "spheres"], viewer.sidechains, setter("sidechains")))
        colorings = ["element", *viewer.coloring_names]
        if len(colorings) > 1:
            form.addRow(
                "color", combo(colorings, "element", lambda c: setattr(viewer, "coloring", None if c == "element" else c))
            )

        # layers
        for layer in ["atoms", "surface", "trails", "ghost", "clashes", *viewer.vector_names, *viewer.point_names]:
            try:
                checked = viewer.is_visible(layer)
            except KeyError:
                continue
            if layer == "ghost" and not checked:
                continue  # no ghost set
            box = QCheckBox(layer, checked=checked)
            box.toggled.connect(lambda on, layer=layer: (viewer.set_visible(layer, on), self.view.request_render()))
            layout.addWidget(box)

        # sizes
        sizes = QFormLayout()
        layout.addLayout(sizes)
        self._slider(sizes, "atom radius", viewer.sizes["radius_scale"], 0.1, 2.0,
                     lambda v: viewer.set_sizes(radius_scale=v))
        self._slider(sizes, "surface opacity", viewer.surface["opacity"], 0.05, 1.0,
                     lambda v: viewer.set_surface(opacity=v))
        sizes.addRow("surface color", combo(["uniform", "atoms"], viewer.surface["colors"],
                                            lambda c: viewer.set_surface(colors=c)))
        for name in viewer.vector_names:
            scale = viewer.vector_scale(name)
            self._slider(sizes, f"{name} length", scale, 0.0, 4.0 * scale,
                         lambda v, name=name: viewer.set_vector_scale(name, v))

        buttons = {"copy camera": camera_json, **(buttons or {})}
        for label, text in buttons.items():
            button = QPushButton(label)
            button.clicked.connect(lambda _=False, text=text: self.copy(text(viewer)))
            layout.addWidget(button)

        self.plots = []
        for name, values in (plots or {}).items():
            xs, ys = values if isinstance(values, tuple) else (self.t[: len(values)], values)
            self.plots.append(Plot(name, xs, ys))
            layout.addWidget(self.plots[-1])
        layout.addStretch()

        splitter = QSplitter()
        splitter.addWidget(self.view)
        splitter.addWidget(panel)
        splitter.setStretchFactor(0, 1)
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(splitter)
        self.resize(1300, 800)
        self.set_frame(viewer.frame)

    def _slider(self, form, label, value, lo, hi, apply, steps=200):
        slider = QSlider(Qt.Orientation.Horizontal, minimum=0, maximum=steps)
        slider.setValue(round((value - lo) / (hi - lo) * steps))
        slider.valueChanged.connect(lambda k: (apply(lo + (hi - lo) * k / steps), self.view.request_render()))
        form.addRow(label, slider)

    def set_scrub_discrete(self, on: bool):
        self.viewer.scrub_discrete = on  # keeps the relative position
        frame = self.viewer.frame
        self.slider.blockSignals(True)
        self.slider.setMaximum(max(self.viewer.num_frames, 1) - 1)
        self.slider.blockSignals(False)
        self.set_frame(frame)

    def current_t(self) -> float:
        frame = self.viewer.frame
        if self.viewer.scrub_discrete:  # discrete states spread evenly over the path
            return float(np.interp(frame / max(self.viewer.num_frames - 1, 1), [0, 1], self.t[[0, -1]]))
        return float(self.t[min(frame, len(self.t) - 1)])

    def set_frame(self, frame: int):
        self.viewer.frame = frame
        if self.slider.value() != frame:
            self.slider.setValue(frame)
        self.frame_label.setText(f"{self.viewer.frame} / {self.slider.maximum()}")
        for plot in self.plots:
            plot.set_x(self.current_t())
        self.view.request_render()

    @staticmethod
    def copy(text: str):
        QApplication.clipboard().setText(text)
        print(text, flush=True)


def camera_json(viewer) -> str:
    return json.dumps({k: (list(v) if isinstance(v, (list, tuple)) else round(v, 3)) for k, v in viewer.camera.items()})


def explore(viewer, **kwargs):
    """Show an Explorer for `viewer` and run the Qt event loop until it closes."""
    app = QApplication.instance() or QApplication([])
    window = Explorer(viewer, **kwargs)
    window.show()
    app.exec()
