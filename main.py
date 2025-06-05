import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
              np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


def cosine_similarity(angles1, angles2):
    dot_product = np.dot(angles1, angles2)
    norm1 = np.linalg.norm(angles1)
    norm2 = np.linalg.norm(angles2)
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot_product / (norm1 * norm2)


def similarity_to_percentage(cos_similarity):
    return (cos_similarity + 1) / 2 * 100


# Точки для расчёта углов
angle_points = [
    (11, 13, 15),  # Левый локоть
    (12, 14, 16),  # Правый локоть
    (23, 25, 27),  # Левое колено
    (24, 26, 28),  # Правое колено
    (14, 12, 24),  # Правое плечо
    (13, 11, 23),  # Левое плечо
    (12, 24, 26),  # Правый таз
    (11, 23, 25),  # Левый таз
]

# Инициализация MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)


def extract_angles_from_frame(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)
    if not results.pose_landmarks:
        return None

    landmarks = results.pose_landmarks.landmark
    pose_coords = np.array([[lm.x, lm.y] for lm in landmarks])

    angles = []
    for i, j, k in angle_points:
        angles.append(calculate_angle(pose_coords[i], pose_coords[j], pose_coords[k]))

    return angles


# Видео вход
cap_ideal = cv2.VideoCapture('source.mp4')
cap_real = cv2.VideoCapture('source.mp4')

angles_ideal_all = []
angles_real_all = []
similarities = []
percentages = []

while cap_ideal.isOpened() and cap_real.isOpened():
    ret_ideal, frame_ideal = cap_ideal.read()
    ret_real, frame_real = cap_real.read()

    if not ret_ideal or not ret_real:
        break

    angles_ideal = extract_angles_from_frame(frame_ideal)
    angles_real = extract_angles_from_frame(frame_real)

    if angles_ideal is None or angles_real is None:
        continue  # Пропускаем кадры без обнаруженных поз

    angles_ideal_all.append(angles_ideal)
    angles_real_all.append(angles_real)

    similarity = cosine_similarity(angles_ideal, angles_real)
    percentage = similarity_to_percentage(similarity)

    similarities.append(similarity)
    percentages.append(percentage)

# Освобождение ресурсов
cap_ideal.release()
cap_real.release()
cv2.destroyAllWindows()

# --- Построение графиков ---

x = list(range(len(similarities)))

# 1. Similarity и Percentage
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(x, similarities, label='Cosine Similarity', color='blue')
plt.title('Cosine Similarity over Time')
plt.xlabel('Frame')
plt.ylabel('Similarity')
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(x, percentages, label='Similarity Percentage', color='green')
plt.title('Similarity Percentage over Time')
plt.xlabel('Frame')
plt.ylabel('%')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 2. Углы
angles_ideal_all = np.array(angles_ideal_all)
angles_real_all = np.array(angles_real_all)

plt.figure(figsize=(14, 8))
for i in range(angles_ideal_all.shape[1]):
    plt.plot(angles_ideal_all[:, i], label=f'Ideal Angle {i+1}', linestyle='--')
    plt.plot(angles_real_all[:, i], label=f'Real Angle {i+1}')
plt.title('Joint Angles Comparison')
plt.xlabel('Frame')
plt.ylabel('Angle (degrees)')
plt.legend(loc='upper right', fontsize='small', ncol=2)
plt.grid(True)
plt.tight_layout()
plt.show()
