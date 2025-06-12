import os
import sys
import importlib.util
import tempfile
from pathlib import Path

# Ensure real numpy and cv2 are used even if earlier tests stubbed them
if 'numpy' in sys.modules:
    del sys.modules['numpy']
if 'cv2' in sys.modules:
    del sys.modules['cv2']

import numpy as np
import cv2

spec = importlib.util.spec_from_file_location('video_utils', Path(__file__).resolve().parents[1] / 'video_utils.py')
video_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(video_utils)
save_video_frames = video_utils.save_video_frames


def create_temp_video(path, frame_count=5, size=(64, 64)):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, 10.0, size)
    for i in range(frame_count):
        img = np.full((size[1], size[0], 3), i*10, dtype=np.uint8)
        out.write(img)
    out.release()


def test_save_video_frames_basic():
    with tempfile.TemporaryDirectory() as temp_dir:
        video_path = os.path.join(temp_dir, 'video.mp4')
        create_temp_video(video_path, frame_count=5)

        output_dir = os.path.join(temp_dir, 'frames')
        saved = save_video_frames(video_path, output_dir, step=2)

        files = sorted(os.listdir(output_dir))
        assert saved == len(files)
        assert saved == 3  # frames 0,2,4
        assert files[0].startswith('frame_000000')
