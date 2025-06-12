import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn

from ai import normalize_skeleton
from format import generate_recommendations


class LSTMClassifier(nn.Module):
    """Простой LSTM-классификатор."""

    def __init__(self, input_size, hidden_size=64, num_layers=1, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])


def extract_skeletons(video_path, max_frames=300):
    """Извлекает последовательность ключевых точек из видео."""
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False)
    cap = cv2.VideoCapture(video_path)
    frames = []

    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if res.pose_landmarks:
            pts = np.array(
                [[lm.x, lm.y] for lm in res.pose_landmarks.landmark],
                dtype=np.float32,
            )
            frames.append(pts)

    cap.release()
    pose.close()

    if not frames:
        raise ValueError("Не удалось извлечь ключевые точки")

    return np.stack(frames)


def dtw_distance(seq1, seq2):
    """Простой подсчет DTW между двумя последовательностями."""
    n, m = len(seq1), len(seq2)
    dtw = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.linalg.norm(seq1[i - 1] - seq2[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    return float(dtw[n, m])


def evaluate_error(reference, sequence):
    """Вычисляет среднее отклонение между двумя последовательностями."""
    T = min(len(reference), len(sequence))
    diff = reference[:T] - sequence[:T]
    return float(np.mean(np.linalg.norm(diff, axis=(1, 2))))


def classify_sequence(sequence, model):
    """Возвращает предсказанный класс для последовательности."""
    model.eval()
    with torch.no_grad():
        inp = torch.from_numpy(sequence.reshape(1, len(sequence), -1)).float()
        out = model(inp)
        return int(out.argmax(dim=1).item())


def run_pipeline(reference_video, target_video, model_path=None, tolerance=10):
    """Полный цикл обработки видео.

    Parameters
    ----------
    reference_video : str
        Path to the reference video.
    target_video : str
        Path to the video to analyse.
    model_path : str, optional
        Path to a trained model.
    tolerance : float, optional
        Allowed deviation in degrees for ``generate_recommendations``.
    """
    ref = normalize_skeleton(extract_skeletons(reference_video))
    seq = normalize_skeleton(extract_skeletons(target_video))

    distance = dtw_distance(ref, seq)
    print(f"DTW distance: {distance:.4f}")

    model = None
    if model_path:
        input_size = ref.shape[1] * ref.shape[2]
        model = LSTMClassifier(input_size=input_size)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        label = classify_sequence(seq, model)
        print(f"Predicted label: {label}")

    error = evaluate_error(ref, seq)
    print(f"Average error: {error:.4f}")

    recs = generate_recommendations(ref[-1], seq[-1], tolerance=tolerance)
    if recs:
        print("Recommendations:")
        for r in recs:
            print("-", r)
    else:
        print("No significant deviations detected.")


if __name__ == "__main__":
    # Пример запуска: python pipeline.py ref.mp4 input.mp4 model.pth
    import sys

    if len(sys.argv) < 3:
        print("Usage: python pipeline.py <reference_video> <target_video> [model] [tolerance]")
        sys.exit(1)

    ref_video = sys.argv[1]
    tar_video = sys.argv[2]
    mdl = sys.argv[3] if len(sys.argv) > 3 else None
    tol = float(sys.argv[4]) if len(sys.argv) > 4 else 10
    run_pipeline(ref_video, tar_video, mdl, tolerance=tol)
