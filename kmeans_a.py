import math
import random

def euclidean(x: list, y: list):
    dist = 0
    for xx, yy in zip(x, y):
        dist += (xx - yy)**2
    return math.sqrt(dist)


def mean(points):
    n = len(points)
    dims = len(points[0])
    m = [0.0 for _ in range(dims)]
    for p in points:
        for i, c in enumerate(p):
            m[i] += c

    for i in range(dims):
        m[i] /= n
    return m


def sum_of_squares(points, centroid):
    s = 0.0
    for p in points:
        s += euclidean(p, centroid)**2
    return s


def cost(clusters, centroids):
    s = 0.0
    for cluster, centroid in zip(clusters, centroids):
        s += sum_of_squares(cluster, centroid)
    return s


def assign_closest(points, centroids):
    k = len(centroids)
    clusters = [[] for _ in range(k)]

    for point in points:
        closest = 0
        min_distance = euclidean(point, centroids[closest])
        for i in range(1, k):
            d = euclidean(point, centroids[i])
            if d < min_distance:
                min_distance = d
                closest = i
        clusters[closest].append(point)

    return clusters
    

def lloyd_iter(points, centroids):
    k = len(centroids)
    clusters = assign_closest(points, centroids)

    # Update
    new_centroids = []
    for i in range(k):
        new_centroids.append(mean(clusters[i]))

    return new_centroids, clusters


def random_centroids(points, k):
    return random.sample(points, k)


def lloyd(points, k, max_iter=300, epsilon=1e-4):
    centroids = random_centroids(points, k)
    clusters = assign_closest(points, centroids)
    wcss = cost(clusters, centroids)

    for iter in range(max_iter):
        new_centroids, new_clusters = lloyd_iter(points, centroids)
        new_wcss = cost(new_clusters, new_centroids)
        if wcss - new_wcss <= epsilon:
            break
        centroids = new_centroids
        clusters = new_clusters
        wcss = new_wcss

    return wcss, clusters, centroids
        

    
