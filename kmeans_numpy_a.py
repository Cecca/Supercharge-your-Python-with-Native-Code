import numpy as np


def assign_closest(points, centroids, points_sq=None):
    n = points.shape[0]
    k = centroids.shape[0]
    assignment = np.zeros(n)

    wcss = 0.0

    for i in range(n):
        closest = 0
        min_distance = np.linalg.norm(points[i] - centroids[closest])
        for c in range(1, k):
            d = np.linalg.norm(points[i] - centroids[c])
            if d < min_distance:
                min_distance = d
                closest = c
        assignment[i] = closest
        wcss += min_distance*min_distance
        
    return assignment, wcss


def random_centroids(points, k, seed=None):
    gen = np.random.default_rng(seed)
    idx = gen.choice(points.shape[0], k, replace=False)
    return points[idx]


def lloyd(points, k, max_iter=300, epsilon=1e-4, seed=None):
    centroids = random_centroids(points, k, seed)
    points_sq = np.einsum("ij,ij->i", points, points)  # constant across iterations
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

