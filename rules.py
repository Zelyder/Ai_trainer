"""Utility rules for validating joint angles."""

from typing import Iterable, List


DEFAULT_RULES = [
    (40, 170, "Слишком согнут левый локоть", "Недостаточно согнут левый локоть"),
    (40, 170, "Слишком согнут правый локоть", "Недостаточно согнут правый локоть"),
    (40, 160, "Слишком согнуто левое колено", "Недостаточно согнуто левое колено"),
    (40, 160, "Слишком согнуто правое колено", "Недостаточно согнуто правое колено"),
]


def check_joint_angle_rules(angles: Iterable[float], rules=DEFAULT_RULES) -> List[str]:
    """Return messages for angles outside the allowed ranges.

    Parameters
    ----------
    angles : iterable of float
        Angles in degrees in order: left elbow, right elbow, left knee, right knee.
    rules : iterable of tuple
        Sequence of (min_angle, max_angle, low_msg, high_msg) for each joint.

    Returns
    -------
    list of str
        Error descriptions for each violated rule.
    """
    msgs = []
    for angle, (mn, mx, low_msg, high_msg) in zip(angles, rules):
        if angle < mn:
            msgs.append(low_msg)
        elif angle > mx:
            msgs.append(high_msg)
    return msgs
