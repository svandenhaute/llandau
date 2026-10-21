import subprocess

import numpy as np
import pytest

from llandau import Viewer
from llandau.video import Video, find_ffmpeg

pytestmark = pytest.mark.skipif(find_ffmpeg() is None, reason="needs ffmpeg")


def test_frames_become_an_mp4(tmp_path):
    viewer = Viewer().load(np.array([[[0.0, 0, 0], [1.0, 0, 0]]]), np.array([8, 1]))
    path = tmp_path / "out.mp4"
    with Video(path, (161, 121), fps=10) as video:  # odd sizes are padded to even
        for _ in range(5):
            video.write(viewer.render_image(161, 121))
    probe = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-show_entries",
                            "stream=width,height,nb_read_frames", "-of", "csv=p=0", str(path)],
                           capture_output=True, text=True)
    if probe.returncode == 0:
        assert probe.stdout.strip() == "162,122,5"
    assert path.stat().st_size > 0


def test_a_failed_video_leaves_no_file(tmp_path):
    path = tmp_path / "out.mp4"
    with pytest.raises(ValueError):
        with Video(path, (16, 16)) as video:
            video.write(np.zeros((8, 8, 4), np.uint8))
    assert not path.exists()
