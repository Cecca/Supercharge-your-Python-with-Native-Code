import csv
import os
import shutil
import time
import urllib.request

import h5py
import numpy as np
import random
from sklearn.cluster import KMeans

import kmeans_a

DATASET_URL = "http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5"
DATASET_PATH = "fashion-mnist-784-euclidean.hdf5"
SAMPLE_SIZE = 1000
SEED = 1234
K = 10
TIMEOUT_S = 60
RESULTS_PATH = "results.csv"


def download_dataset(url=DATASET_URL, path=DATASET_PATH):
    if os.path.exists(path):
        return path
    print(f"Downloading {url} ...")
    # The host (Cloudflare) rejects urllib's default user-agent with 403.
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(path, "wb") as out:
        shutil.copyfileobj(resp, out)
    print(f"Saved to {path}")
    return path


def load_data(path=DATASET_PATH, seed=SEED):
    with h5py.File(path, "r") as f:
        data = f["train"][:]
    # Shuffle once so size-n samples are nested prefixes of the same ordering.
    order = np.random.default_rng(seed).permutation(len(data))
    return data[order]


def dataset_sizes(total, start=SAMPLE_SIZE):
    """Sizes from `start`, doubling, up to and including the full dataset size."""
    if total <= start:
        return [total]
    sizes = []
    n = start
    while n < total:
        sizes.append(n)
        n *= 2
    sizes.append(total)
    return sizes


def load_completed(path=RESULTS_PATH):
    """Map (implementation, n, k) -> recorded time for runs already in the CSV."""
    completed = {}
    if not os.path.exists(path):
        return completed
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["implementation"], int(row["n"]), int(row["k"]))
            completed[key] = float(row["time_seconds"])  # keep latest recorded time
    return completed


def record_result(implementation, n, dim, wcss, elapsed):
    print(f"{implementation}: k={K} n={n} dim={dim} wcss={wcss:.2f} time={elapsed:.4f}s")
    write_header = not os.path.exists(RESULTS_PATH)
    with open(RESULTS_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["implementation", "n", "k", "dim", "wcss", "time_seconds"])
        writer.writerow([implementation, n, K, dim, wcss, elapsed])


def run_kmeans_a(data):
    points = data.tolist()
    # kmeans_a.random_centroids uses the global random module; seed it for reproducibility.
    random.seed(SEED)
    start = time.perf_counter()
    wcss, _clusters, _centroids = kmeans_a.lloyd(points, K)
    elapsed = time.perf_counter() - start
    record_result("kmeans_a", len(points), len(points[0]), wcss, elapsed)
    return elapsed


def run_sklearn(data):
    # n_init=1 matches kmeans_a, which runs from a single random initialization.
    model = KMeans(n_clusters=K, init="random", n_init=1, random_state=SEED)
    start = time.perf_counter()
    model.fit(data)
    elapsed = time.perf_counter() - start
    # inertia_ is the within-cluster sum of squares, same metric as kmeans_a's wcss.
    record_result("sklearn", data.shape[0], data.shape[1], model.inertia_, elapsed)
    return elapsed


def main():
    download_dataset()
    data = load_data()

    completed = load_completed()
    runners = {"kmeans_a": run_kmeans_a, "sklearn": run_sklearn}
    active = set(runners)

    for n in dataset_sizes(len(data)):
        if not active:
            break
        subset = data[:n]
        for name in list(runners):  # stable order, only run still-active ones
            if name not in active:
                continue
            key = (name, n, K)
            if key in completed:
                elapsed = completed[key]
                print(f"{name}: n={n} already in {RESULTS_PATH} ({elapsed:.4f}s), skipping")
            else:
                elapsed = runners[name](subset)
            if elapsed > TIMEOUT_S:
                print(f"{name} exceeded {TIMEOUT_S}s at n={n}; stopping it.")
                active.discard(name)


if __name__ == "__main__":
    main()
