"""The explorer: a native window with the viewer and a panel of controls and plots.

    from riemann_viewer import Viewer
    from riemann_viewer.explorer import explore

    viewer = Viewer().load(positions, numbers, roles=roles)
    viewer.add_vectors("tangent", tangents)
    explore(viewer, plots={"energy": energy})   # blocks until the window closes

Mouse: drag rotates, right drag (or shift + drag) pans, the wheel scrubs frames,
ctrl + wheel or a pinch zooms, a double click resets the camera. Keys: space plays,
, and . step frames, D toggles the discrete states, 0 resets the camera, Escape
closes. No Qt needed: the window is part of riemann_viewer (winit + egui).
"""
from __future__ import annotations

import numpy as np

__all__ = ["explore"]


def _curve(name, values, t):
    """(name, xs, values): values per frame, or (xs, values) on the scale of t."""
    if isinstance(values, tuple):
        xs, values = values
    else:
        values = np.asarray(values, dtype=np.float64)
        xs = t[: len(values)] if len(values) <= len(t) else np.arange(len(values))
    xs = np.asarray(xs, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if len(xs) != len(values):
        raise ValueError(f"plot {name!r}: {len(xs)} x values for {len(values)} values")
    return name, xs.tolist(), values.tolist()


def explore(viewer, plots: dict | None = None, t=None, info: str | None = None,
            buttons: dict | None = None, title: str = "riemann_viewer", screenshot: str | None = None):
    """Show the viewer in a window until it closes.

    plots       name -> values per frame, or name -> (xs, values) on the scale of `t`
    t           the parameter of each path frame (default: the frame index)
    info        a line of text above the controls
    buttons     label -> function of the camera dict returning text to copy (and print)
    screenshot  write one frame of the window to this PNG file and close (for tests)
    """
    frames = max(viewer.num_path_frames, 1)
    t = np.arange(frames, dtype=np.float64) if t is None else np.asarray(t, dtype=np.float64)
    curves = [_curve(name, values, t) for name, values in (plots or {}).items()]
    viewer.explore(
        plots=curves,
        t=t.tolist(),
        info=info,
        buttons=list((buttons or {}).items()),
        title=title,
        screenshot=screenshot,
    )
