import numpy as np


def calculate_angle(a, b, c):
    """Return the joint angle ABC in degrees using the dot product."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return 0.0
    cos_angle = np.dot(ba, bc) / denom
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    return float(angle)


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
