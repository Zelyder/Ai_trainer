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


def movement_accuracy(coord_error, angle_error, coord_max=1.0, angle_max=180.0,
                      coord_weight=0.5, angle_weight=0.5):
    """Compute final movement accuracy metric.

    The coordinate and angle errors are normalized to their respective maximum
    values. The weighted average error is then subtracted from 1 to obtain an
    accuracy value in the range ``[0, 1]``.

    Parameters
    ----------
    coord_error : float
        Deviation of coordinates from reference in the same units as
        ``coord_max``.
    angle_error : float
        Deviation of joint angles in degrees.
    coord_max : float, optional
        Maximum possible coordinate error. Defaults to ``1.0``.
    angle_max : float, optional
        Maximum possible angle error. Defaults to ``180.0``.
    coord_weight : float, optional
        Relative importance of coordinate accuracy.
    angle_weight : float, optional
        Relative importance of angle accuracy.

    Returns
    -------
    float
        Normalized accuracy from ``0`` (worst) to ``1`` (best).
    """

    if coord_max <= 0 or angle_max <= 0:
        raise ValueError("Maximum errors must be positive")

    total_weight = coord_weight + angle_weight
    if total_weight == 0:
        raise ValueError("Weights must sum to a positive value")

    coord_norm = min(max(coord_error / coord_max, 0.0), 1.0)
    angle_norm = min(max(angle_error / angle_max, 0.0), 1.0)

    weighted_error = ((coord_weight * coord_norm) +
                      (angle_weight * angle_norm)) / total_weight

    accuracy = 1.0 - weighted_error
    return max(0.0, min(1.0, accuracy))


# Пример использования (требуются массивы позы с индексами как в angle_points)
# ideal_pose = [...]  # список координат эталонной позы
# real_pose = [...]   # список координат реальной позы
# recommendations = generate_recommendations(ideal_pose, real_pose)
# for rec in recommendations:
#     print(rec)
