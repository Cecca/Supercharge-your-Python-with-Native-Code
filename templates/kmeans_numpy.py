import numpy as np


def assign_closest(points, centroids, points_sq=None):
    # Vectorized over all points and centroids at once via
    # ||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2, then pick the nearest centroid.
    if points_sq is None:
        points_sq = np.linalg.norm(points, axis=1)**2
    c_sq = np.linalg.norm(centroids, axis=1)**2
    d = points_sq[:, None] - 2.0 * (points @ centroids.T) + c_sq[None, :]
    np.maximum(d, 0.0, out=d)  # rounding can push exact zeros slightly negative

    assignment = np.argmin(d, axis=1)
    wcss = float(np.min(d, axis=1).sum(dtype=np.float64))
    return assignment, wcss


def random_centroids(points, k, seed=None):
    gen = np.random.default_rng(seed)
    idx = gen.choice(points.shape[0], k, replace=False)
    return points[idx]


def lloyd(points, k, max_iter=300, epsilon=1e-4, seed=None):
    centroids = random_centroids(points, k, seed)
    points_sq = np.linalg.norm(points, axis=1)**2  # constant across iterations
    prev_wcss = np.inf

    for _ in range(max_iter):
        assignment, wcss = assign_closest(points, centroids, points_sq)
        if prev_wcss - wcss < epsilon * prev_wcss:  # relative WCSS decrease
            return assignment, wcss
        prev_wcss = wcss

        new_centroids = np.zeros_like(centroids)
        for i in range(k):
            members = points[assignment == i]
            new_centroids[i] = members.mean(axis=0) if len(members) else centroids[i]
        centroids = new_centroids

    return assignment, wcss
