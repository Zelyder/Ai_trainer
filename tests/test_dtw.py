import numpy as np
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('skeleton_dtw', Path(__file__).resolve().parents[1] / 'skeleton_dtw.py')
skeleton_dtw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(skeleton_dtw)
dtw_skeletons = skeleton_dtw.dtw_skeletons


def make_skeleton(shift=0.0):
    pts = np.zeros((17, 2), dtype=np.float32)
    pts[7] = [0 + shift, 0]
    pts[5] = [0 + shift, 1]
    pts[9] = [1 + shift, 0]

    pts[8] = [0 + shift, 0]
    pts[6] = [-1 + shift, 0]
    pts[10] = [1 + shift, 0]

    pts[13] = [0 + shift, 0]
    pts[11] = [0 + shift, 1]
    pts[15] = [0 + shift, -1]

    pts[14] = [0 + shift, 0]
    pts[12] = [1 + shift, 0]
    pts[16] = [0 + shift, 1]
    return pts


def test_dtw_identity():
    seq1 = [make_skeleton() for _ in range(3)]
    seq2 = [make_skeleton() for _ in range(3)]
    dist, path = dtw_skeletons(seq1, seq2)
    assert dist == 0.0
    assert path == [(0, 0), (1, 1), (2, 2)]


def test_dtw_shift():
    seq1 = [make_skeleton() for _ in range(3)]
    seq2 = [make_skeleton(0.1) for _ in range(3)]
    dist, _ = dtw_skeletons(seq1, seq2, w_ang=0)
    assert abs(dist - 0.21176470588235302) < 1e-6


