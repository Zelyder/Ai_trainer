import argparse
from pathlib import Path
import numpy as np

from ai import load_and_clean


def calc_angles_ntu(pts: np.ndarray) -> np.ndarray:
    """Calculate 4 angles for NTU pose format."""
    def ang(a, b, c):
        v1, v2 = a - b, c - b
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

    idxs = [(5, 7, 9), (6, 8, 10), (11, 13, 15), (12, 14, 16)]
    return np.array([ang(pts[i], pts[j], pts[k]) for i, j, k in idxs], dtype=np.float32)


def build_feature_sequences(X: np.ndarray) -> np.ndarray:
    """Combine normalized coordinates and computed angles for each frame."""
    features = []
    for sample in X:
        angles_seq = np.array([calc_angles_ntu(frame) for frame in sample], dtype=np.float32)
        coords = sample.reshape(sample.shape[0], -1)
        seq = np.concatenate([coords, angles_seq], axis=1)
        features.append(seq)
    return np.stack(features)


def process_dataset(input_pkl: Path, output_npz: Path, max_frames: int = 100, max_samples: int = 1000) -> None:
    X, y = load_and_clean(str(input_pkl), max_frames=max_frames, max_samples=max_samples)
    feats = build_feature_sequences(X)
    np.savez_compressed(output_npz, X=feats, y=y)
    print(f"Saved {feats.shape[0]} samples to {output_npz}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare pose sequences for training")
    parser.add_argument("input_pkl", type=Path, help="Path to raw dataset pickle")
    parser.add_argument("output_npz", type=Path, help="Where to save processed data")
    parser.add_argument("--max_frames", type=int, default=100, help="Number of frames per sequence")
    parser.add_argument("--max_samples", type=int, default=1000, help="Limit number of samples")
    args = parser.parse_args()

    process_dataset(args.input_pkl, args.output_npz, args.max_frames, args.max_samples)
