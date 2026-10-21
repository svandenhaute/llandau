"""The explorer: a native window with the viewer and a panel of controls and plots.

    from llandau import Viewer
    from llandau.explorer import explore

    viewer = Viewer().load(positions, numbers, roles=roles)
    viewer.add_vectors("tangent", tangents)
    explore(viewer, plots={"energy": energy})   # blocks until the window closes

Mouse: drag rotates, right drag (or shift + drag) pans, the wheel scrubs frames
(faster when scrolling fast and for long trajectories; shift: frame by frame),
ctrl + wheel or a pinch zooms, a double click resets the camera. Every control
has a key, hinted beside it in the panel; ? shows them all:

    Space P  play          ← → , .  frames (shift: 5%)   Home End  first, last frame
    D  discrete states     + −  zoom      0  reset the camera
    T  light / dark        L  lit / flat  S  atoms / protein (shift: side chains)
    C  next coloring       1-9  layers    ⌘C / ctrl C  copy the camera
    G  large plots         H  hide the panel    F  full screen    R  record
    Esc  close the keys, leave full screen, close

To present: F, H, and G to show the plots large under the view with readable
axes. No Qt needed: the window is part of llandau (winit + egui).

Record (or R) keeps what you do in the view; Stop renders it offscreen into an
MP4 at a fixed frame rate, without the panel (needs ffmpeg: pip install
'llandau[video]').
"""
from __future__ import annotations

import itertools
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
    record      the record button's video file (default: llandau-<date>-<time>.mp4)
    record_size its size in pixels (default: the 3D view's), record_fps its frame rate
    """
    if plots is not None or t is not None:
        viewer.set_plots(plots or {}, t)
    viewer.explore(
        info=info,
        buttons=list((buttons or {}).items()),
        title=title,
        screenshot=screenshot,
        recorder=_recorder(record, record_fps),
        record_size=record_size,
        record_fps=record_fps,
    )
