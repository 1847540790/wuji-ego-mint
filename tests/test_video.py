from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from mint.inference.video import read_video


def write_test_video(path: Path, frames: int = 8, fps: float = 8.0) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (32, 24),
    )
    assert writer.isOpened()
    for index in range(frames):
        frame = np.full((24, 32, 3), index * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_video_decode_enforces_frame_and_fps_limits(tmp_path: Path) -> None:
    source = tmp_path / "sample.avi"
    write_test_video(source)

    video = read_video(source, max_frames=3, target_fps=4.0)

    assert video.frames_rgb.shape == (3, 24, 32, 3)
    assert video.fps == pytest.approx(4.0)
    assert video.source_fps == pytest.approx(8.0)
    assert video.source_frames == 8


def test_video_decode_rejects_missing_input(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Video does not exist"):
        read_video(tmp_path / "missing.mp4")
