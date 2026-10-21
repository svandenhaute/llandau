"""The explorer: a native window with the viewer and a panel of controls and plots.

    from llandau import Viewer
    from llandau.explorer import explore

    viewer = Viewer().load(positions, numbers, roles=roles)
    viewer.add_vectors("tangent", tangents)
    explore(viewer, plots={"energy": energy})   # blocks until the window closes

Mouse: drag rotates, right drag (or shift + drag) pans, the wheel scrubs frames
(faster when scrolling fast and for long trajectories; shift: frame by frame),
ctrl + wheel or a pinch zooms, a double click resets the camera. Keys: space plays,
, and . step frames (shift: 5% of them), Home and End go to the first and last
frame, D toggles the discrete states, 0 resets the camera, R starts and stops
recording, Escape closes. No Qt needed: the window is part of llandau (winit + egui).

Record (or R) keeps what you do in the view; Stop renders it offscreen into an
MP4 at a fixed frame rate, without the panel (needs ffmpeg: pip install
'llandau[video]').
"""
from __future__ import annotations

import itertools
import time
from pathlib import Path

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


def _recorder(path, fps):
    """The record button's video writers: (width, height) -> Video at path (then
    path-2, path-3, ...), by default llandau-<date>-<time>.mp4; None without ffmpeg."""
    from .video import Video, find_ffmpeg
    if find_ffmpeg() is None:
        return None
    count = itertools.count(1)

    def start(width, height):
        k = next(count)
        if path is None:
            target = Path(time.strftime("llandau-%Y%m%d-%H%M%S.mp4"))
        else:
            target = Path(path)
            target = target.with_name(f"{target.stem}-{k}{target.suffix}") if k > 1 else target
        return Video(target, (width, height), fps=fps)
    return start


def explore(viewer, plots: dict | None = None, t=None, info: str | None = None,
            buttons: dict | None = None, title: str = "llandau", screenshot: str | None = None,
            record: str | None = None, record_size: tuple[int, int] | None = None, record_fps: float = 60):
    """Show the viewer in a window until it closes.

    plots       name -> values per frame, or name -> (xs, values) on the scale of `t`
    t           the parameter of each path frame (default: the frame index)
    info        a line of text above the controls
    buttons     label -> function of the camera dict returning text to copy (and print)
    screenshot  write one frame of the window to this PNG file and close (for tests)
    record      the record button's video file (default: llandau-<date>-<time>.mp4)
    record_size its size in pixels (default: the 3D view's), record_fps its frame rate
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
        recorder=_recorder(record, record_fps),
        record_size=record_size,
        record_fps=record_fps,
    )
