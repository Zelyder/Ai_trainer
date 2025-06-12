import numpy as np
from dataclasses import dataclass
from typing import Any

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


@dataclass
class FrameFeatures:
    """Container for features extracted from a single pose frame."""

    angles: Any
    normalized_points: Any


def normalize_pose(pose,
                   hip_left: int = 23,
                   hip_right: int = 24,
                   shoulder_left: int = 11,
                   shoulder_right: int = 12):
    """Normalize coordinates relative to pelvis and scale by shoulder width."""

    arr = np.asarray(pose, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("pose must have shape (V, 2)")

    pelvis = (arr[hip_left] + arr[hip_right]) / 2.0
    arr = arr - pelvis

    scale = np.linalg.norm(arr[shoulder_left] - arr[shoulder_right])
    if scale > 0:
        arr = arr / scale

    return arr


def build_frame_features(pose) -> FrameFeatures:
    """Return joint angles and normalized coordinates for a pose frame."""

    normalized = normalize_pose(pose)
    angles = np.array(
        [calculate_angle(normalized[i], normalized[j], normalized[k])
         for i, j, k in ANGLE_POINTS],
        dtype=np.float32,
    )
    return FrameFeatures(angles=angles, normalized_points=normalized)


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
