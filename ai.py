import numpy as np
import pickle
import threading
from sklearn.preprocessing import MinMaxScaler


def normalize_skeleton(skel):
    T, V, C = skel.shape
    center = (skel[:, 23, :] + skel[:, 24, :]) / 2  # точки бедер
    skel = skel - center[:, None, :]
    # масштабируем по максимальному размаху
    flat = skel.reshape(-1, C)
    dists = np.linalg.norm(flat[:, None, :] - flat[None, :, :], axis=-1)
    maxd = np.nanmax(dists)
    if maxd > 0:
        skel /= maxd
    return skel


def load_and_clean(path_pkl, max_frames=100, max_samples=1000):
    with open(path_pkl, 'rb') as f:
        raw = pickle.load(f)

    annotations = raw['annotations']
    X, y = [], []
    dropped = 0

    for i, sample in enumerate(annotations):
        keypoints = sample.get('keypoint', None)
        label = sample.get('label', None)
        shape = sample.get('img_shape', None)

        if keypoints is None or label is None or shape is None:
            dropped += 1
            continue

        keypoints = np.array(keypoints, dtype=np.float32)

        # Исправляем форму: (1, T, V, 2) -> (T, V, 2)
        if keypoints.ndim == 4 and keypoints.shape[0] == 1:
            keypoints = np.squeeze(keypoints, axis=0)

        if keypoints.ndim != 3 or keypoints.shape[2] != 2:
            print(f"[{i}] Пропущен: неправильная форма keypoints: {keypoints.shape}")
            dropped += 1
            continue

        h, w = shape
        if h == 0 or w == 0:
            print(f"[{i}] Пропущен: нулевые размеры изображения: {shape}")
            dropped += 1
            continue

        # Нормализация координат
        keypoints[:, :, 0] /= w
        keypoints[:, :, 1] /= h

        if np.isnan(keypoints).any():
            print(f"[{i}] Пропущен: есть NaN в keypoints")
            dropped += 1
            continue

        # Padding/обрезка по длине
        if keypoints.shape[0] < max_frames:
            pad_len = max_frames - keypoints.shape[0]
            pad = np.zeros((pad_len, keypoints.shape[1], 2), dtype=np.float32)
            keypoints = np.vstack([keypoints, pad])
        else:
            keypoints = keypoints[:max_frames]

        X.append(keypoints)
        y.append(label)

        if len(X) >= max_samples:
            break

    if len(X) == 0:
        raise ValueError("Нет подходящих примеров после очистки")

    X = np.stack(X)
    y = np.array(y)

    print("Загружено:", len(X), "Пропущено:", dropped)
    return X, y


# вызов функции
path = "ntu60_hrnet.pkl"

X, y = load_and_clean(path, max_samples=500)
print("Форма X:", X.shape)
print("Форма y:", y.shape)

# === 4. ОСНОВНОЙ РЕАЛЬНОЙ ВРЕМЕНИ МОДУЛЬ ===
import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import pyttsx3


# -- модели --
class LSTMClassifier(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return torch.softmax(self.fc(hn[-1]), dim=1)


class RecommendationNet(nn.Module):
    def __init__(self, input_size=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )

    def forward(self, x): return self.net(x)


# -- утилиты --
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
engine = pyttsx3.init()
engine.setProperty('rate', 150)
spoken = [False] * 5


def compute_reference_angles(X):
    all_angles = []
    for sample in X:
        angles_seq = []
        for frame in sample:
            angles_seq.append(calc_angles_ntu(frame))
        all_angles.append(angles_seq)
    return np.mean(all_angles, axis=0)  # (frames, 4)


def extract_keypoints(frame, pose_model=None):
    if pose_model is None:
        pose_model = pose
    res = pose_model.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not res.pose_landmarks:
        return None
    return np.array([[lm.x, lm.y] for lm in res.pose_landmarks.landmark], dtype=np.float32)


def calc_angles(pts):
    def ang(a, b, c):
        v1, v2 = a - b, c - b
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        return np.degrees(np.arccos(np.clip(cos, -1, 1)))

    idxs = [(11, 13, 15), (12, 14, 16), (23, 25, 27), (24, 26, 28)]
    return np.array([ang(pts[i], pts[j], pts[k]) for i, j, k in idxs], dtype=np.float32)


# Индексы для углов: (плечо-локоть-запястье), (бедро-колено-лодыжка)
ntu_idxs = [(5, 7, 9),  # левый локоть
            (6, 8, 10),  # правый локоть
            (11, 13, 15),  # левое колено
            (12, 14, 16)]  # правое колено


def calc_angles_ntu(pts):
    def ang(a, b, c):
        v1, v2 = a - b, c - b
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

    idxs = [(5, 7, 9), (6, 8, 10), (11, 13, 15), (12, 14, 16)]
    return np.array([ang(pts[i], pts[j], pts[k]) for i, j, k in idxs], dtype=np.float32)


def draw_info(frame, angs, ref, recs):
    for i, (a, r) in enumerate(zip(angs, ref)):
        dev = abs(a - r)
        cv2.putText(frame, f'#{i + 1}: {a:.1f}Δ{dev:.1f}',
                    (10, 30 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    for i, rec in enumerate(recs):
        cv2.putText(frame, f'R{i + 1}:{rec:.2f}',
                    (10, 150 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    return frame


def speak_async(msg):
    def worker():
        engine.say(msg)
        engine.runAndWait()

    threading.Thread(target=worker, daemon=True).start()


def speak(angs, ref, recs, thr=15):
    msgs = ["Выпрямьте левый локоть", "Выпрямьте правый локоть",
            "Сгибайте левое колено", "Сгибайте правое колено"]
    for i, dev in enumerate(np.abs(angs - ref)):
        if dev > thr and not spoken[i]:
            speak_async(msgs[i])
            spoken[i] = True
        elif dev <= thr:
            spoken[i] = False  # разрешаем повторную озвучку при следующем нарушении


# -- инициализация моделей --
clf = LSTMClassifier(input_size=4)
rec_net = RecommendationNet(input_size=4)


# Эталон (усреднённая последовательность по всем образцам, 4 угла)
def extract_reference_from_video(video_path, max_frames=100):
    cap = cv2.VideoCapture(video_path)
    angles_seq = []
    with mp_pose.Pose(static_image_mode=False) as pose_ref:
        while len(angles_seq) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            pts = extract_keypoints(frame, pose_ref)
            if pts is not None and len(pts) >= 17:
                angles = calc_angles_ntu(pts)
                angles_seq.append(angles)
    cap.release()
    if len(angles_seq) == 0:
        raise ValueError("Не удалось извлечь ключевые точки из видео")
    return np.array(angles_seq)  # shape = (T, 4)


reference = extract_reference_from_video("source.mp4", max_frames=100)

# === Запуск обработки с веб-камеры ===
cap = cv2.VideoCapture(0)
seq = []
max_len = reference.shape[0]

while True:
    ret, frame = cap.read()
    if not ret: break

    pts = extract_keypoints(frame)
    if pts is not None and len(pts) >= 17:
        angs = calc_angles_ntu(pts)
        seq.append(angs)
        if len(seq) > max_len:
            seq.pop(0)
        ref_ang = reference[len(seq) - 1]
        inp = torch.tensor(np.abs(angs - ref_ang), dtype=torch.float32)
        recs = rec_net(inp).detach().numpy()
        speak(angs, ref_ang, recs)
        frame = draw_info(frame, angs, ref_ang, recs)

    cv2.imshow('Feedback', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
