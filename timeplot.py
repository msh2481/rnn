import argparse

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot validation score against runtime from opt_results.csv."
    )
    parser.add_argument(
        "--input",
        default="opt_results.csv",
        help="Path to CSV with name, r2, and time columns.",
    )
    parser.add_argument(
        "--output",
        default="timeplot.png",
        help="Optional output image path. If omitted, shows the plot interactively.",
    )
    return parser.parse_args()


def extract_model_type(name: str) -> str:
    return name.split("(", 1)[0]


def main():
    args = parse_args()
    df = pd.read_csv(args.input)

    required_columns = {"name", "r2", "time"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns in {args.input}: {missing}")

    df = df.copy()
    df["model_type"] = df["name"].map(extract_model_type)
    df = df[df["time"] > 0].reset_index(drop=True)
    if df.empty:
        raise ValueError("No rows with positive time values were found.")

    model_types = sorted(df["model_type"].unique())
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(9, 6))
    for color_ix, model_type in enumerate(model_types):
        subset = df[df["model_type"] == model_type]
        ax.scatter(
            subset["time"],
            subset["r2"],
            label=model_type,
            color=cmap(color_ix % 10),
            alpha=0.75,
            s=36,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Time per run (seconds, log scale)")
    ax.set_ylabel("R^2")
    ax.set_title("Validation Score vs Runtime")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(title="Model type")
    fig.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=150)
        print(f"Saved plot to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
