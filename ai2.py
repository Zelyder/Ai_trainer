import cv2
import numpy as np
import mediapipe as mp
import tkinter as tk
from tkinter import filedialog
import torch
import torch.nn as nn
import threading
import pyttsx3
from PIL import Image, ImageDraw, ImageFont
import format

try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except Exception:
    device = "cpu"

# === Модели ===
class RecommendationNet(nn.Module):
    def __init__(self, input_size=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )
    def forward(self, x): return self.net(x)

# === Углы ===
def calc_angles_ntu(pts):
    def ang(a, b, c):
        v1, v2 = a - b, c - b
        cos = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2) + 1e-6)
        return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

    idxs = [(5, 7, 9), (6, 8, 10), (11, 13, 15), (12, 14, 16)]
    return np.array([ang(pts[i], pts[j], pts[k]) for i, j, k in idxs], dtype=np.float32)

# === Поза и озвучка ===
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
engine = pyttsx3.init()
engine.setProperty('rate', 150)
spoken_hints = set()
last_recs = np.array([0.0] * 5, dtype=np.float32)

def speak_async(msg):
    def worker():
        engine.say(msg)
        engine.runAndWait()
    threading.Thread(target=worker, daemon=True).start()

def speak(user_pts, ref_pts):
    """Generate and voice hints using ``format.generate_recommendations``."""
    hints = format.generate_recommendations(ref_pts, user_pts)
    global spoken_hints
    new_spoken = set()
    for msg in hints:
        if msg not in spoken_hints:
            speak_async(msg)
        new_spoken.add(msg)
    spoken_hints = new_spoken
    return hints


def infer_async(model, input_tensor, callback):
    def worker():
        with torch.no_grad():
            output = model(input_tensor)
        callback(output.cpu())

    threading.Thread(target=worker, daemon=True).start()

# === Ключевые точки ===
def extract_keypoints(frame, pose_model=None):
    if pose_model is None:
        pose_model = pose
    res = pose_model.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not res.pose_landmarks:
        return None
    return np.array([[lm.x, lm.y] for lm in res.pose_landmarks.landmark], dtype=np.float32)

# === Извлечение эталона ===
def extract_reference_from_video(video_path, max_frames=1000):
    cap = cv2.VideoCapture(video_path)
    angles_seq = []
    frames_seq = []
    keypoints_seq = []
    with mp_pose.Pose(static_image_mode=False) as pose_ref:
        while len(angles_seq) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            pts = extract_keypoints(frame, pose_ref)
            if pts is not None and len(pts) >= 17:
                angles = calc_angles_ntu(pts)
                angles_seq.append(angles)
                frames_seq.append(cv2.resize(frame, (320, 240)))
                keypoints_seq.append(pts)
    cap.release()
    if len(angles_seq) == 0:
        raise ValueError("Не удалось извлечь ключевые точки из видео")
    return np.array(angles_seq), frames_seq, keypoints_seq

# === Отображение инфо ===
def draw_info(frame, angs, ref, recs):
    for i, (a, r) in enumerate(zip(angs, ref)):
        dev = abs(a - r)
        cv2.putText(frame, f'#{i+1}: {a:.1f} Δ{dev:.1f}',
                    (10, 30 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    for i, rec in enumerate(recs):
        cv2.putText(frame, f'R{i+1}:{rec:.2f}',
                    (10, 150 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    return frame

def draw_text_pil(image, text, pos, font_size=20, color=(255, 255, 255)):
    """Рисует кириллический текст поверх изображения OpenCV с помощью PIL"""
    # Преобразуем изображение в формат PIL
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        # Используем системный шрифт с поддержкой кириллицы (Windows/Linux)
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        # Запасной шрифт (можно указать путь к TTF-файлу вручную)
        font = ImageFont.load_default()
    draw.text(pos, text, font=font, fill=color)
    # Обратно в OpenCV
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def draw_two_skeletons(frame, ref_pts, user_pts, connections=None,
                       ref_color=(255, 0, 0), user_color=(0, 0, 255),
                       radius=3, thickness=2):
    """Overlay two pose skeletons on ``frame``.

    Parameters
    ----------
    frame : ndarray
        Image in BGR format where skeletons will be drawn.
    ref_pts : array-like shape (N, 2)
        Normalized landmark coordinates of the reference pose.
    user_pts : array-like shape (N, 2)
        Normalized landmark coordinates of the user's pose.
    connections : iterable of tuple[int, int], optional
        Pairs of landmark indices describing the skeleton edges.  When
        ``None`` (default), ``mediapipe`` pose connections are used if
        available.
    ref_color : tuple[int, int, int]
        BGR color for the reference skeleton (default blue).
    user_color : tuple[int, int, int]
        BGR color for the user skeleton (default red).
    radius : int
        Radius of drawn keypoints.
    thickness : int
        Thickness of connection lines.
    """

    h, w = frame.shape[:2]

    if connections is None:
        connections = getattr(mp_pose, 'POSE_CONNECTIONS', [])

    def draw_one(points, color):
        for i, j in connections:
            if i >= len(points) or j >= len(points):
                continue
            p1 = (int(points[i][0] * w), int(points[i][1] * h))
            p2 = (int(points[j][0] * w), int(points[j][1] * h))
            cv2.line(frame, p1, p2, color, thickness)
        for pt in points:
            x, y = int(pt[0] * w), int(pt[1] * h)
            cv2.circle(frame, (x, y), radius, color, -1)

    if ref_pts is not None:
        draw_one(ref_pts, ref_color)
    if user_pts is not None:
        draw_one(user_pts, user_color)

    return frame

# === Интерфейс загрузки ===
def load_video_and_run():
    root = tk.Tk()
    root.withdraw()
    video_path = filedialog.askopenfilename(title="Выберите эталонное видео", filetypes=[("Video files", "*.mp4")])
    if not video_path:
        print("Видео не выбрано.")
        return

    print("Извлекаем эталонные углы...")
    reference_angles, reference_frames, reference_points = extract_reference_from_video(video_path)
    print(f"Готово: {len(reference_angles)} кадров.")

    run_camera_with_reference(reference_angles, reference_frames, reference_points)

# === Главный цикл ===
ANGLE_TOLERANCE = 15
def run_camera_with_reference(reference, ref_frames, ref_points):
    global ANGLE_TOLERANCE
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('AI trainer', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('AI trainer', 1280, 720)
    seq = []
    net = RecommendationNet(input_size=4).to(device)
    ref_len = len(reference)
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pts = extract_keypoints(frame)
        if pts is not None and len(pts) >= 17:
            angs = calc_angles_ntu(pts)
            seq.append(angs)
            if len(seq) > ref_len:
                seq.pop(0)

            ref_ang = reference[idx % ref_len]
            ref_frame = ref_frames[idx % ref_len]
            ref_pts = ref_points[idx % ref_len]
            idx += 1

            inp = torch.tensor(np.abs(angs - ref_ang),
                               dtype=torch.float32, device=device)

            def cb(out):
                recs = out.numpy()
                global last_recs
                last_recs = recs

            infer_async(net, inp, cb)
            hints = speak(pts, ref_pts)
            recs = last_recs
            frame = draw_info(frame, angs, ref_ang, recs)
            for i, msg in enumerate(hints):
                frame = draw_text_pil(frame, msg, (10, 180 + i * 25), font_size=20, color=(255, 0, 0))
            similarity = max(0.0, 1.0 - np.mean(np.abs(angs - ref_ang)) / 90.0)

            # Отображаем схожесть и текущий порог
            frame = draw_text_pil(frame, f'Сходство: {similarity*100:.1f}%', (10, 310), font_size=20)
            frame = draw_text_pil(frame, f'Допустимое отклонение: {ANGLE_TOLERANCE}°', (10, 340), font_size=20)

            frame = cv2.resize(frame, (480, 360))
            ref_frame = cv2.resize(ref_frame, (480, 360))
            combo = np.hstack([frame, ref_frame])
            cv2.imshow('AI trainer', combo)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('+') or key == ord('='):
            ANGLE_TOLERANCE += 1
        elif key == ord('-') and ANGLE_TOLERANCE > 1:
            ANGLE_TOLERANCE -= 1

    cap.release()
    cv2.destroyAllWindows()


# === Запуск ===
if __name__ == "__main__":
    load_video_and_run()
