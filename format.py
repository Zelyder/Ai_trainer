import numpy as np


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
    angle_points = [
        (11, 13, 15),  # Левый локоть
        (12, 14, 16),  # Правый локоть
        (23, 25, 27),  # Левое колено
        (24, 26, 28)   # Правое колено
    ]

    recommendations = []
    for i, j, k in angle_points:
        ideal_angle = calculate_angle(ideal_pose[i], ideal_pose[j], ideal_pose[k])
        real_angle = calculate_angle(real_pose[i], real_pose[j], real_pose[k])
        if abs(ideal_angle - real_angle) > 10:
            recommendations.append(
                f"Корректируйте угол между точками {i}-{j}-{k}. "
                f"Эталонный: {ideal_angle:.2f}, ваш: {real_angle:.2f}"
            )
    return recommendations


# Пример использования (требуются массивы позы с индексами как в angle_points)
# ideal_pose = [...]  # список координат эталонной позы
# real_pose = [...]   # список координат реальной позы
# recommendations = generate_recommendations(ideal_pose, real_pose)
# for rec in recommendations:
#     print(rec)
