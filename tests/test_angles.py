import sys
import types
import math


class FakeArray(list):
    def __sub__(self, other):
        return FakeArray([a - b for a, b in zip(self, other)])

    def __rsub__(self, other):
        return FakeArray([b - a for a, b in zip(self, other)])


class FakeNumpy(types.ModuleType):
    float32 = float
    pi = math.pi

    def array(self, obj, dtype=None):
        return FakeArray(obj)

    def arctan2(self, y, x):
        return math.atan2(y, x)

    def abs(self, x):
        return abs(x)

    def degrees(self, x):
        return math.degrees(x)

    def clip(self, x, a, b):
        return max(a, min(b, x))

    def dot(self, a, b):
        return sum(x * y for x, y in zip(a, b))

    class linalg:
        @staticmethod
        def norm(a):
            return math.sqrt(sum(x * x for x in a))

    def arccos(self, x):
        return math.acos(x)

    def zeros(self, shape, dtype=float):
        return [FakeArray([0.0 for _ in range(shape[1])]) for _ in range(shape[0])]


# inject fake numpy before importing modules that require it
sys.modules.setdefault('numpy', FakeNumpy('numpy'))

import importlib.util
from pathlib import Path

# Import functions under test
format_spec = importlib.util.spec_from_file_location('format', Path(__file__).resolve().parents[1] / 'format.py')
format = importlib.util.module_from_spec(format_spec)
format_spec.loader.exec_module(format)
calculate_angle = format.calculate_angle

ai2_path = Path(__file__).resolve().parents[1] / 'ai2.py'
ai2_spec = importlib.util.spec_from_file_location('ai2', ai2_path)
ai2 = importlib.util.module_from_spec(ai2_spec)
# stub additional deps for ai2
for name in ['cv2', 'tkinter', 'tkinter.filedialog', 'pyttsx3', 'PIL', 'PIL.Image', 'PIL.ImageDraw', 'PIL.ImageFont']:
    sys.modules.setdefault(name, types.ModuleType(name))

pyttsx3_stub = sys.modules.setdefault('pyttsx3', types.ModuleType('pyttsx3'))
def _init():
    class Engine:
        def setProperty(self, *a, **k):
            pass
    return Engine()
pyttsx3_stub.init = _init
torch_module = types.ModuleType('torch')
torch_nn = types.ModuleType('torch.nn')
class DummyModule: pass
torch_nn.Module = DummyModule
torch_nn.Linear = lambda *a, **k: None
torch_nn.ReLU = lambda *a, **k: None
torch_nn.Sequential = lambda *a, **k: None
torch_module.nn = torch_nn
sys.modules.setdefault('torch', torch_module)
sys.modules.setdefault('torch.nn', torch_nn)
mediapipe_stub = types.SimpleNamespace(solutions=types.SimpleNamespace(pose=types.SimpleNamespace(Pose=lambda *a, **k: None)))
sys.modules.setdefault('mediapipe', mediapipe_stub)
ai2_spec.loader.exec_module(ai2)


def test_calculate_angle_basic():
    assert calculate_angle([1, 0], [0, 0], [0, 1]) == 90.0
    assert calculate_angle([-1, 0], [0, 0], [1, 0]) == 180.0


def test_calc_angles_ntu():
    np = sys.modules['numpy']
    pts = np.zeros((17, 2))

    pts[7] = np.array([0, 0])
    pts[5] = np.array([0, 1])
    pts[9] = np.array([1, 0])

    pts[8] = np.array([0, 0])
    pts[6] = np.array([-1, 0])
    pts[10] = np.array([1, 0])

    pts[13] = np.array([0, 0])
    pts[11] = np.array([0, 1])
    pts[15] = np.array([0, -1])

    pts[14] = np.array([0, 0])
    pts[12] = np.array([1, 0])
    pts[16] = np.array([0, 1])

    angles = ai2.calc_angles_ntu(pts)
    expected = [90, 180, 180, 90]
    assert all(abs(a - b) < 1.0 for a, b in zip(angles, expected))


rules_spec = importlib.util.spec_from_file_location('rules', Path(__file__).resolve().parents[1] / 'rules.py')
rules = importlib.util.module_from_spec(rules_spec)
rules_spec.loader.exec_module(rules)


def test_check_joint_angle_rules():
    angles = [180, 175, 30, 170]
    msgs = rules.check_joint_angle_rules(angles)
    expected = [
        "Недостаточно согнут левый локоть",
        "Недостаточно согнут правый локоть",
        "Слишком согнуто левое колено",
        "Недостаточно согнуто правое колено",
    ]
    assert msgs == expected


def test_generate_recommendations_extended():
    np = sys.modules['numpy']
    base_ideal = np.zeros((29, 2))
    base_real = np.zeros((29, 2))
    for i, j, k in format.ANGLE_POINTS:
        base_ideal[i] = np.array([1, 0])
        base_ideal[j] = np.array([0, 0])
        base_ideal[k] = np.array([0, 1])
        base_real[i] = np.array([1, 0])
        base_real[j] = np.array([0, 0])
        base_real[k] = np.array([0, 1])

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

    for triple in format.ANGLE_POINTS:
        ideal = [row[:] for row in base_ideal]
        real = [row[:] for row in base_real]
        i, j, k = triple
        real[k] = np.array([1, 0])  # distort one joint

        recs = format.generate_recommendations(ideal, real)
        # Expect at least one hint about the changed joint
        hint = messages[triple]
        assert any(hint in r for r in recs)

