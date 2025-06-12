import math


def _euclidean(p, q):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p, q)))


def calc_angles_ntu(pts):
    """Calculate four NTU angles for a skeleton."""
    def angle(a, b, c):
        v1 = (a[0] - b[0], a[1] - b[1])
        v2 = (c[0] - b[0], c[1] - b[1])
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        norm = (_euclidean(v1, (0, 0)) * _euclidean(v2, (0, 0))) + 1e-6
        cos = max(-1.0, min(1.0, dot / norm))
        return math.degrees(math.acos(cos))

    idxs = [(5, 7, 9), (6, 8, 10), (11, 13, 15), (12, 14, 16)]
    return [angle(pts[i], pts[j], pts[k]) for i, j, k in idxs]


def skeleton_metric(a, b, w_euc=1.0, w_ang=1.0):
    n = len(a)
    euc = sum(_euclidean(a[i], b[i]) for i in range(n)) / n
    ang_a = calc_angles_ntu(a)
    ang_b = calc_angles_ntu(b)
    ang = sum(abs(x - y) for x, y in zip(ang_a, ang_b)) / len(ang_a)
    return w_euc * euc + w_ang * ang


def dtw_skeletons(seq1, seq2, w_euc=1.0, w_ang=1.0):
    n, m = len(seq1), len(seq2)
    D = [[float('inf')] * (m + 1) for _ in range(n + 1)]
    D[0][0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = skeleton_metric(seq1[i - 1], seq2[j - 1], w_euc, w_ang)
            D[i][j] = cost + min(D[i - 1][j], D[i][j - 1], D[i - 1][j - 1])

    # path reconstruction
    i, j = n, m
    path = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        choices = [
            (D[i - 1][j - 1], 'diag'),
            (D[i - 1][j], 'up'),
            (D[i][j - 1], 'left'),
        ]
        step = min(choices, key=lambda x: x[0])[1]
        if step == 'diag':
            i -= 1
            j -= 1
        elif step == 'up':
            i -= 1
        else:  # left
            j -= 1
    path.reverse()

    return D[n][m], path
