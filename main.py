import argparse
import cProfile
import csv
import os
import pstats
import shutil
import time
import urllib.request

import h5py
import numpy as np
import random
from sklearn.cluster import KMeans

import kmeans_a
import kmeans_numpy
import kmeans_numba

DATASETS = {
    "fashion-mnist": {
        "url": "http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5",
        "path": "fashion-mnist-784-euclidean.hdf5",
    },
    "glove-100": {
        "url": "http://ann-benchmarks.com/glove-100-angular.hdf5",
        "path": "glove-100-angular.hdf5",
    },
}
DEFAULT_DATASET = "fashion-mnist"
SAMPLE_SIZE = 1000
SEED = 1234
K = 10
TIMEOUT_S = 60
RESULTS_PATH = "results.csv"


def download_dataset(name=DEFAULT_DATASET):
    url, path = DATASETS[name]["url"], DATASETS[name]["path"]
    if os.path.exists(path):
        return path
    print(f"Downloading {url} ...")
    # The host (Cloudflare) rejects urllib's default user-agent with 403.
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(path, "wb") as out:
        shutil.copyfileobj(resp, out)
    print(f"Saved to {path}")
    return path


def load_data(name=DEFAULT_DATASET, seed=SEED):
    with h5py.File(DATASETS[name]["path"], "r") as f:
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


RESULTS_HEADER = ["dataset", "implementation", "n", "k", "dim", "seed", "wcss", "time_seconds"]


def migrate_results(path=None):
    """Add a `dataset` column to legacy rows (assumed to be fashion-mnist)."""
    if path is None:
        path = RESULTS_PATH
    if not os.path.exists(path):
        return
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None or "dataset" in header:
            return
        rows = list(reader)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset"] + header)
        for row in rows:
            writer.writerow([DEFAULT_DATASET] + row)
    print(f"Migrated {path}: added dataset column (filled as {DEFAULT_DATASET}).")


def load_completed(path=RESULTS_PATH):
    """Map (dataset, implementation, n, k, seed) -> recorded time for runs in the CSV."""
    completed = {}
    if not os.path.exists(path):
        return completed
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            dataset = row.get("dataset") or DEFAULT_DATASET
            key = (dataset, row["implementation"], int(row["n"]), int(row["k"]), int(row["seed"]))
            completed[key] = float(row["time_seconds"])  # keep latest recorded time
    return completed


def record_result(dataset, implementation, n, dim, wcss, elapsed):
    print(
        f"{implementation}: dataset={dataset} k={K} n={n} dim={dim} seed={SEED} "
        f"wcss={wcss:.2f} time={elapsed:.4f}s"
    )
    write_header = not os.path.exists(RESULTS_PATH)
    with open(RESULTS_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(RESULTS_HEADER)
        writer.writerow([dataset, implementation, n, K, dim, SEED, wcss, elapsed])


def run_kmeans_a(data, dataset):
    points = data.tolist()
    # kmeans_a.random_centroids uses the global random module; seed it for reproducibility.
    random.seed(SEED)
    start = time.perf_counter()
    wcss, _clusters, _centroids = kmeans_a.lloyd(points, K)
    elapsed = time.perf_counter() - start
    record_result(dataset, "kmeans_a", len(points), len(points[0]), wcss, elapsed)
    return elapsed


def run_kmeans_numpy(data, dataset):
    start = time.perf_counter()
    _assignment, wcss = kmeans_numpy.lloyd(data, K, seed=SEED)
    elapsed = time.perf_counter() - start
    record_result(dataset, "kmeans_numpy", data.shape[0], data.shape[1], wcss, elapsed)
    return elapsed


def run_kmeans_numba(data, dataset):
    # Warm up the JIT on a tiny slice so compilation stays out of the timed run.
    kmeans_numba.lloyd(data[: 2 * K], K, max_iter=1, seed=SEED)
    start = time.perf_counter()
    _assignment, wcss = kmeans_numba.lloyd(data, K, seed=SEED)
    elapsed = time.perf_counter() - start
    record_result(dataset, "kmeans_numba", data.shape[0], data.shape[1], wcss, elapsed)
    return elapsed


def run_sklearn(data, dataset):
    # n_init=1 matches kmeans_a, which runs from a single random initialization.
    model = KMeans(n_clusters=K, init="random", n_init=1, random_state=SEED)
    start = time.perf_counter()
    model.fit(data)
    elapsed = time.perf_counter() - start
    # inertia_ is the within-cluster sum of squares, same metric as kmeans_a's wcss.
    record_result(dataset, "sklearn", data.shape[0], data.shape[1], model.inertia_, elapsed)
    return elapsed


RUNNERS = {
    "kmeans_a": run_kmeans_a,
    "kmeans_numpy": run_kmeans_numpy,
    "kmeans_numba": run_kmeans_numba,
    "sklearn": run_sklearn,
}


def _load_subset(dataset, n):
    download_dataset(dataset)
    data = load_data(dataset)
    if n > len(data):
        raise SystemExit(f"requested n={n} exceeds dataset size {len(data)}")
    return data[:n]


def profile_run(algorithm, n, dataset=DEFAULT_DATASET, output=None, top=30):
    subset = _load_subset(dataset, n)
    runner = RUNNERS[algorithm]
    profiler = cProfile.Profile()
    profiler.enable()
    runner(subset, dataset)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumulative")
    stats.print_stats(top)
    if output:
        stats.dump_stats(output)
        print(f"Profile written to {output}")


def memray_run(algorithm, n, output, dataset=DEFAULT_DATASET, native=False, follow_fork=False):
    try:
        import memray
    except ImportError as e:
        raise SystemExit("memray is not installed. Try: pip install memray") from e
    if os.path.exists(output):
        raise SystemExit(f"{output} already exists; pick a new path or delete it.")
    subset = _load_subset(dataset, n)
    runner = RUNNERS[algorithm]
    with memray.Tracker(output, native_traces=native, follow_fork=follow_fork):
        runner(subset, dataset)
    print(f"Allocation profile written to {output}")
    print(f"View with: memray flamegraph {output}   (or: memray tree {output})")


def main():
    migrate_results()
    completed = load_completed()

    for dataset in DATASETS:
        download_dataset(dataset)
        data = load_data(dataset)
        active = set(RUNNERS)

        for n in dataset_sizes(len(data)):
            if not active:
                break
            subset = data[:n]
            for name in list(RUNNERS):  # stable order, only run still-active ones
                if name not in active:
                    continue
                key = (dataset, name, n, K, SEED)
                if key in completed:
                    elapsed = completed[key]
                    print(
                        f"{name}: dataset={dataset} n={n} already in {RESULTS_PATH} "
                        f"({elapsed:.4f}s), skipping"
                    )
                else:
                    elapsed = RUNNERS[name](subset, dataset)
                if elapsed > TIMEOUT_S:
                    print(
                        f"{name} exceeded {TIMEOUT_S}s at n={n} on {dataset}; "
                        f"stopping it for this dataset."
                    )
                    active.discard(name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark k-means implementations.")
    parser.add_argument(
        "--profile",
        metavar="ALGORITHM",
        choices=sorted(RUNNERS),
        help="Run a single algorithm under cProfile instead of the full benchmark.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=SAMPLE_SIZE,
        help="Number of points to use when profiling (default: %(default)s).",
    )
    parser.add_argument(
        "--profile-output",
        metavar="PATH",
        help="If set with --profile, dump the pstats profile to this file.",
    )
    parser.add_argument(
        "--profile-top",
        type=int,
        default=30,
        help="Number of rows to print from the profile (default: %(default)s).",
    )
    parser.add_argument(
        "--memray",
        metavar="ALGORITHM",
        choices=sorted(RUNNERS),
        help="Run a single algorithm under memray allocation tracking.",
    )
    parser.add_argument(
        "--memray-output",
        metavar="PATH",
        default="memray.bin",
        help="Output file for the memray capture (default: %(default)s).",
    )
    parser.add_argument(
        "--memray-native",
        action="store_true",
        help="Capture native stack traces (slower, but resolves NumPy/Numba frames).",
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        default=DEFAULT_DATASET,
        help="Dataset to use for --profile/--memray (default: %(default)s).",
    )
    args = parser.parse_args()

    if args.profile and args.memray:
        parser.error("--profile and --memray are mutually exclusive.")

    if args.profile:
        profile_run(
            args.profile, args.n,
            dataset=args.dataset, output=args.profile_output, top=args.profile_top,
        )
    elif args.memray:
        memray_run(
            args.memray, args.n, output=args.memray_output,
            dataset=args.dataset, native=args.memray_native,
        )
    else:
        main()
