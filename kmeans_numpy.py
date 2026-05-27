import numpy as np

def euclidean_squared(x, y):
    diff = x - y
    return np.dot(diff, diff)


def assign_closest(points, centroids):
    k = centroids.shape[0]
    n = points.shape[0]
    assignment = np.zeros(n)

    wcss = 0.0
    for i, point in enumerate(points):
        closest = 0
        min_distance = euclidean_squared(point, centroids[closest])
        for j in range(1, k):
            d = euclidean_squared(point, centroids[j])
            if d < min_distance:
                min_distance = d
                closest = j
        assignment[i] = closest
        wcss += min_distance

    return assignment, wcss


def random_centroids(points, k, seed=None):
    gen = np.random.default_rng(seed)
    idx = gen.choice(points.shape[0], k, replace=False)
    return points[idx]


def lloyd(points, k, max_iter=20, epsilon=0.0, seed=None):
    centroids = random_centroids(points, k, seed)
    prev_wcss = np.inf

    for _ in range(max_iter):
        assignment, wcss = assign_closest(points, centroids)
        if prev_wcss - wcss < epsilon:
            return assignment, wcss
        prev_wcss = wcss

        centroids = np.zeros_like(centroids)
        for i in range(k):
            centroids[i, :] = np.mean(points[assignment == i], axis=0)

    return assignment, wcss


 
