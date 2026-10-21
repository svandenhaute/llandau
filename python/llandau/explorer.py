"""The explorer: a native window with the viewer and a panel of controls and plots.

    from llandau import Viewer
    from llandau.explorer import explore

    viewer = Viewer().load(positions, numbers, roles=roles)
    viewer.add_vectors("tangent", tangents)
    explore(viewer, plots={"energy": energy})   # blocks until the window closes

Mouse: drag rotates, right drag (or shift + drag) pans, the wheel scrubs frames
(faster when scrolling fast and for long trajectories; shift: frame by frame),
ctrl + wheel or a pinch zooms, a double click resets the camera. Every control
has a key, hinted beside it in the panel; ? in the window shows every key.

To present: F (full screen), H (hide the panel) and G show the plots large
under the view with readable axes. No Qt needed: the window is part of llandau
(winit + egui).

Record (or R) keeps what you do in the view as keyframes of its state (the
camera, the frame, every setting of the panel) in a JSON file, which a deck
plays on a step with live(play="file.json"). Recording to an .mp4 renders a
video offscreen instead, at a fixed frame rate, without the panel (needs
ffmpeg: pip install 'llandau[video]'). Copy (or the copy state button) copies
the view's state as a dict, for live(view=...).
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path


__all__ = ["explore"]


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


def _keyframes(path):
    """The record button's keyframe writer: samples (time, state) -> the JSON file written,
    by default llandau-<date>-<time>.json."""
    def save(samples):
        target = Path(path) if path is not None else Path(time.strftime("llandau-%Y%m%d-%H%M%S.json"))
        target.write_text(json.dumps([{"t": t, **state} for t, state in samples], indent=1))
        return str(target)
    return save


def explore(viewer, plots: dict | None = None, t=None, info: str | None = None,
            buttons: dict | None = None, title: str = "llandau", screenshot: str | None = None,
            record: str | None = None, record_size: tuple[int, int] | None = None, record_fps: float = 60):
    """Show the viewer in a window until it closes.

    plots       name -> values per frame, or name -> (xs, values) on the scale of `t`
                (as viewer.set_plots; default: the viewer's)
    t           the parameter of each path frame (default: the frame index)
    info        a line of text above the controls
    buttons     label -> function of the camera dict returning text to copy (and print)
    screenshot  write one frame of the window to this PNG file and close (for tests)
    record      the record button's file: keyframes of the view's state as JSON (default:
                llandau-<date>-<time>.json), or a video when it ends in .mp4
    record_size the video's size in pixels (default: the 3D view's), record_fps its frame rate
    """
    if plots is not None or t is not None:
        viewer.set_plots(plots or {}, t)
    video = record is not None and record.lower().endswith(".mp4")
    viewer.explore(
        info=info,
        buttons=list((buttons or {}).items()),
        title=title,
        screenshot=screenshot,
        recorder=_recorder(record, record_fps) if video else None,
        record_size=record_size,
        record_fps=record_fps,
        keyframes=None if video else _keyframes(record),
    )
