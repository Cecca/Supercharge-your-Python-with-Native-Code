import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True, cache=True)
def _assign(points, centroids):
    # Fused: each point streams its features once per centroid, keeping only the
    # running best squared distance -- the n x k distance matrix is never built.
    # Each iteration writes distinct slots (no shared reduction), so prange is a
    # clean parallel map; the wcss sum is done by the caller.
    n, dim = points.shape
    k = centroids.shape[0]
    assignment = np.empty(n, dtype=np.int64)
    min_dist = np.empty(n, dtype=np.float64)
    for i in prange(n):
        best = 0
        best_d = 0.0
        for d in range(dim):
            diff = points[i, d] - centroids[0, d]
            best_d += diff * diff
        for j in range(1, k):
            dist = 0.0
            for d in range(dim):
                diff = points[i, d] - centroids[j, d]
                dist += diff * diff
            if dist < best_d:
                best_d = dist
                best = j
        assignment[i] = best
        min_dist[i] = best_d
    return assignment, min_dist


@njit(fastmath=True, cache=True)
def _update(points, assignment, centroids):
    # One O(n*dim) pass to accumulate per-cluster sums; cheap next to _assign.
    n, dim = points.shape
    k = centroids.shape[0]
    sums = np.zeros((k, dim), dtype=np.float64)
    counts = np.zeros(k, dtype=np.int64)
    for i in range(n):
        c = assignment[i]
        counts[c] += 1
        for d in range(dim):
            sums[c, d] += points[i, d]

    new_centroids = centroids.copy()  # keeps the old centroid for empty clusters
    for j in range(k):
        if counts[j] > 0:
            for d in range(dim):
                new_centroids[j, d] = sums[j, d] / counts[j]
    return new_centroids


def random_centroids(points, k, seed=None):
    gen = np.random.default_rng(seed)
    idx = gen.choice(points.shape[0], k, replace=False)
    return points[idx].copy()


def lloyd(points, k, max_iter=300, epsilon=1e-4, seed=None):
    points = np.ascontiguousarray(points)
    centroids = random_centroids(points, k, seed)
    prev_wcss = np.inf

    assignment = np.empty(points.shape[0], dtype=np.int64)
    wcss = np.inf
    for _ in range(max_iter):
        assignment, min_dist = _assign(points, centroids)
        wcss = float(min_dist.sum())
        if prev_wcss - wcss < epsilon * prev_wcss:  # relative WCSS decrease
            return assignment, wcss
        prev_wcss = wcss
        centroids = _update(points, assignment, centroids)

    return assignment, wcss
