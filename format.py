import numpy as np

# Pairs of keypoint indices used for angle-based feedback. Indices follow
# the MediaPipe pose specification.
ANGLE_POINTS = [
    (11, 13, 15),  # Левый локоть
    (12, 14, 16),  # Правый локоть
    (23, 25, 27),  # Левое колено
    (24, 26, 28),  # Правое колено
    (14, 12, 24),  # Правое плечо
    (13, 11, 23),  # Левое плечо
    (12, 24, 26),  # Правый таз
    (11, 23, 25),  # Левый таз
]


def calculate_angle(a, b, c):
    """Calculate the angle (in degrees) formed by three points."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
              np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


def generate_recommendations(ideal_pose, real_pose):
    """Return textual feedback for each joint angle that differs noticeably."""

    messages = {
        (11, 13, 15): "Выпрямьте левый локоть",
        (12, 14, 16): "Выпрямьте правый локоть",
        (23, 25, 27): "Сгибайте левое колено",
        (24, 26, 28): "Сгибайте правое колено",
        (14, 12, 24): "Держите правое плечо ровно",
        (13, 11, 23): "Держите левое плечо ровно",
        (12, 24, 26): "Не отклоняйте правый таз",
        (11, 23, 25): "Не отклоняйте левый таз",
    }

    recommendations = []
    for i, j, k in ANGLE_POINTS:
        ideal_angle = calculate_angle(ideal_pose[i], ideal_pose[j], ideal_pose[k])
        real_angle = calculate_angle(real_pose[i], real_pose[j], real_pose[k])
        if abs(ideal_angle - real_angle) > 10:
            msg = messages.get((i, j, k), f"Корректируйте угол {i}-{j}-{k}")
            recommendations.append(
                f"{msg}. Эталонный: {ideal_angle:.2f}, ваш: {real_angle:.2f}"
            )
    return recommendations


# Пример использования (требуются массивы позы с индексами как в angle_points)
# ideal_pose = [...]  # список координат эталонной позы
# real_pose = [...]   # список координат реальной позы
# recommendations = generate_recommendations(ideal_pose, real_pose)
# for rec in recommendations:
#     print(rec)
