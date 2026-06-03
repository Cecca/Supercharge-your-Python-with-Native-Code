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


def plot(df, output, implementations=None):
    if implementations is None or len(implementations) == 0:
        implementations = sorted(df["implementation"].unique())
    print(df)

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

    # Move the legend from the figure margin into the right side of the first panel.
    if grid._legend is not None:
        grid._legend.remove()
    first_ax = grid.axes.flat[0]
    handles, labels = first_ax.get_legend_handles_labels()
    first_ax.legend(handles, labels, title="implementation", loc="center right")
    grid.figure.suptitle("k-means runtime by implementation")
    grid.figure.tight_layout()
    grid.savefig(output, dpi=150)
    print(f"Wrote {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot k-means benchmark results.")
    parser.add_argument("--input", default=RESULTS_PATH, help="CSV file to read (default: %(default)s).")
    parser.add_argument("--output", default=OUTPUT_PATH, help="Image file to write (default: %(default)s).")
    parser.add_argument("--show", action="store_true", help="Also display the figure interactively.")
    parser.add_argument("--implementations", help="The implementations to show", nargs="*")
    args = parser.parse_args()

    implementations = args.implementations

    df = load_results(args.input)
    if df.is_empty():
        raise SystemExit(f"No rows found in {args.input}")
    plot(df, args.output, implementations)
    if args.show:
        plt.show()
