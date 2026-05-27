import csv
import os
import shutil
import time
import urllib.request

import h5py
import random

import kmeans_a

DATASET_URL = "http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5"
DATASET_PATH = "fashion-mnist-784-euclidean.hdf5"
SAMPLE_SIZE = 1000
SEED = 1234
K = 10
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


def load_sample(path=DATASET_PATH, sample_size=SAMPLE_SIZE, seed=SEED):
    with h5py.File(path, "r") as f:
        data = f["train"][:]
    points = data.tolist()
    rng = random.Random(seed)
    return rng.sample(points, sample_size)


def main():
    download_dataset()
    points = load_sample()

    start = time.perf_counter()
    wcss, clusters, centroids = kmeans_a.lloyd(points, K)
    elapsed = time.perf_counter() - start

    print(f"k={K} n={len(points)} dim={len(points[0])} wcss={wcss:.2f} time={elapsed:.4f}s")

    write_header = not os.path.exists(RESULTS_PATH)
    with open(RESULTS_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["implementation", "n", "k", "dim", "wcss", "time_seconds"])
        writer.writerow(["kmeans_a", len(points), K, len(points[0]), wcss, elapsed])


if __name__ == "__main__":
    main()
