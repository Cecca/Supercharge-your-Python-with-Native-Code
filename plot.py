import argparse
import csv
from collections import defaultdict

import matplotlib.pyplot as plt

RESULTS_PATH = "results.csv"
OUTPUT_PATH = "performance.png"

IMPL_STYLE = {
    "kmeans_a":     {"color": "#d62728", "marker": "o"},
    "kmeans_numpy": {"color": "#1f77b4", "marker": "s"},
    "kmeans_numba": {"color": "#2ca02c", "marker": "^"},
    "sklearn":      {"color": "#9467bd", "marker": "D"},
}


def load_results(path):
    # (dataset, implementation) -> list of (n, time_seconds)
    series = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            series[(row["dataset"], row["implementation"])].append(
                (int(row["n"]), float(row["time_seconds"]))
            )
    for key in series:
        series[key].sort()
    return series


def plot(series, output):
    datasets = sorted({d for d, _ in series})
    implementations = sorted({i for _, i in series})

    fig, axes = plt.subplots(
        1, len(datasets), figsize=(6 * len(datasets), 5), sharey=True, squeeze=False,
    )
    axes = axes[0]

    for ax, dataset in zip(axes, datasets):
        for impl in implementations:
            points = series.get((dataset, impl), [])
            if not points:
                continue
            xs, ys = zip(*points)
            style = IMPL_STYLE.get(impl, {})
            ax.plot(xs, ys, label=impl, linewidth=1.5, markersize=6, **style)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("n (points)")
        ax.set_title(dataset)
        ax.grid(True, which="both", linestyle=":", alpha=0.5)

    axes[0].set_ylabel("time (s)")
    axes[-1].legend(loc="best", frameon=True)
    fig.suptitle("k-means runtime by implementation")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Wrote {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot k-means benchmark results.")
    parser.add_argument("--input", default=RESULTS_PATH, help="CSV file to read (default: %(default)s).")
    parser.add_argument("--output", default=OUTPUT_PATH, help="Image file to write (default: %(default)s).")
    parser.add_argument("--show", action="store_true", help="Also display the figure interactively.")
    args = parser.parse_args()

    series = load_results(args.input)
    if not series:
        raise SystemExit(f"No rows found in {args.input}")
    plot(series, args.output)
    if args.show:
        plt.show()
