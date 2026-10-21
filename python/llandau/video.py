"""Videos: RGBA frames encoded to MP4 (H.264) by ffmpeg, streamed as they come.

    with Video("turn.mp4", (1920, 1080), fps=60) as video:
        for angle in angles:
            video.write(viewer.render_image(1920, 1080))

ffmpeg is the LLANDAU_FFMPEG executable, else the one on the PATH, else the one
of imageio-ffmpeg (pip install 'llandau[video]'). The explorer's record button
writes its videos with this.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

__all__ = ["Video", "find_ffmpeg"]


def find_ffmpeg() -> str | None:
    """The ffmpeg executable, or None."""
    path = os.environ.get("LLANDAU_FFMPEG") or shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        return None


class Video:
    """An MP4 file written frame by frame: (height, width, 4 or 3) uint8 arrays of the
    given size (width, height). quality is x264's crf: lower is better and larger
    (18 looks lossless; 23 is ffmpeg's default)."""

    def __init__(self, path, size, fps: float = 60, quality: int = 18, ffmpeg: str | None = None):
        ffmpeg = ffmpeg or find_ffmpeg()
        if ffmpeg is None:
            raise RuntimeError("writing videos needs ffmpeg: pip install 'llandau[video]', or put ffmpeg on the PATH")
        self.path = Path(path)
        self.size = (int(size[0]), int(size[1]))
        width, height = self.size
        self.process = subprocess.Popen([
            ffmpeg, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{width}x{height}", "-r", f"{fps}", "-i", "-",
            # yuv420p (what every player plays) needs an even size; BT.709 as HD players assume
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2,scale=out_color_matrix=bt709:out_range=tv",
            "-c:v", "libx264", "-preset", "slow", "-crf", str(quality), "-pix_fmt", "yuv420p",
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "iec61966-2-1",
            "-movflags", "+faststart", str(self.path),
        ], stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame):
        frame = np.asarray(frame, dtype=np.uint8)
        width, height = self.size
        if frame.shape[:2] != (height, width) or frame.ndim != 3 or frame.shape[2] not in (3, 4):
            raise ValueError(f"frame: expected ({height}, {width}, 4 or 3), got {frame.shape}")
        if frame.shape[2] == 3:
            frame = np.dstack([frame, np.full((height, width), 255, np.uint8)])
        try:
            self.process.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError:
            self._fail()

    def close(self) -> str:
        """Finish the file; returns its path."""
        if self.process.stdin and not self.process.stdin.closed:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
        if self.process.wait() != 0:
            self._fail()
        return str(self.path)

    def abort(self):
        """Stop without a file."""
        self.process.kill()
        self.process.wait()
        self.path.unlink(missing_ok=True)

    def _fail(self):
        error = self.process.stderr.read().decode(errors="replace").strip() if self.process.stderr else ""
        self.abort()
        raise RuntimeError(f"ffmpeg failed writing {self.path}: {error or 'no output'}")

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        if kind is None:
            self.close()
        else:
            self.abort()
