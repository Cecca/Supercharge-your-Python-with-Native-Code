import argparse

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

RESULTS_PATH = "results.csv"
OUTPUT_PATH = "performance.png"

def load_results(path):
    # Average the (possibly multiple) runs for each (dataset, implementation, n).
    return (
        pl.read_csv(path)
        .group_by("dataset", "implementation", "n")
        .agg(pl.col("time_seconds").mean())
        .sort("dataset", "implementation", "n")
    )


def plot(df, output):
    implementations = sorted(df["implementation"].unique())

    sns.set_theme(style="whitegrid")
    grid = sns.relplot(
        data=df.to_pandas(),
        x="n",
        y="time_seconds",
        hue="implementation",
        style="implementation",
        col="dataset",
        kind="line",
        # palette=palette,
        markers=True,
        hue_order=implementations,
        style_order=implementations,
        markersize=8,
        linewidth=1.5,
        facet_kws={"sharey": True},
    )

    grid.set(xscale="log", yscale="log")
    grid.set_axis_labels("n (points)", "time (s)")
    grid.set_titles("{col_name}")
    grid.figure.suptitle("k-means runtime by implementation")
    grid.figure.tight_layout()
    grid.savefig(output, dpi=150)
    print(f"Wrote {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot k-means benchmark results.")
    parser.add_argument("--input", default=RESULTS_PATH, help="CSV file to read (default: %(default)s).")
    parser.add_argument("--output", default=OUTPUT_PATH, help="Image file to write (default: %(default)s).")
    parser.add_argument("--show", action="store_true", help="Also display the figure interactively.")
    args = parser.parse_args()

    df = load_results(args.input)
    if df.is_empty():
        raise SystemExit(f"No rows found in {args.input}")
    plot(df, args.output)
    if args.show:
        plt.show()
