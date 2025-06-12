"""Utility for generating textual advice based on detected errors."""

from typing import Iterable, List


# Mapping of known error codes to human-readable advice
_ERROR_ADVICE = {
    'back_bend': 'Следите за спиной — слишком сильный наклон.',
    'left_elbow': 'Выпрямите левый локоть.',
    'right_elbow': 'Выпрямите правый локоть.',
    'left_knee': 'Сгибайте левое колено.',
    'right_knee': 'Сгибайте правое колено.',
}


def generate_advice(errors: Iterable[str]) -> List[str]:
    """Return user advice for a sequence of detected error codes.

    Parameters
    ----------
    errors : Iterable[str]
        Identifiers of detected problems.

    Returns
    -------
    List[str]
        List of advice strings corresponding to the errors.
    """
    advice = []
    for code in errors:
        msg = _ERROR_ADVICE.get(code)
        if msg:
            advice.append(msg)
        else:
            advice.append(f'Проверьте элемент: {code}')
    return advice

