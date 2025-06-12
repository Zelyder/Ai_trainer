import numpy as np
import matplotlib.pyplot as plt


def dtw_matrix_with_path(A, B, dist=None, visualize=True):
    """Compute DTW matrix and optimal path between sequences A and B.

    Parameters
    ----------
    A : Sequence of numeric vectors
        First sequence.
    B : Sequence of numeric vectors
        Second sequence.
    dist : callable, optional
        Distance function between elements. Defaults to Euclidean norm.
    visualize : bool, optional
        If True, display the cost matrix and optimal path using matplotlib.

    Returns
    -------
    tuple of (numpy.ndarray, list)
        The DTW cost matrix and the optimal warping path as list of (i, j)
        index pairs.
    """
    A = np.asarray(A)
    B = np.asarray(B)
    n, m = len(A), len(B)
    if dist is None:
        dist = lambda x, y: np.linalg.norm(x - y)

    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = dist(A[i - 1], B[j - 1])
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

    dtw_matrix = cost[1:, 1:]

    i, j = n, m
    path = [(i - 1, j - 1)]
    while i > 1 or j > 1:
        directions = [
            (cost[i - 1, j], i - 1, j),
            (cost[i, j - 1], i, j - 1),
            (cost[i - 1, j - 1], i - 1, j - 1),
        ]
        step_cost, i, j = min(directions, key=lambda x: x[0])
        path.append((i - 1, j - 1))
    path.reverse()

    if visualize:
        plt.imshow(dtw_matrix, origin="lower", cmap="viridis", interpolation="nearest")
        y, x = zip(*path)
        plt.plot(x, y, color="red")
        plt.title("DTW Cost Matrix")
        plt.xlabel("Sequence B")
        plt.ylabel("Sequence A")
        plt.colorbar(label="Cost")
        plt.show()

    return dtw_matrix, path

