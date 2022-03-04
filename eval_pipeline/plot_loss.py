from __future__ import annotations
import argparse
import sys

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

size_dict = {
    "gpt2": 124_000_000,
    "gpt2-medium": 355_000_000,
    "gpt2-large": 774_000_000,
    "gpt2-xl": 1_500_000_000,
    # GPT-3 sizes are based on https://blog.eleuther.ai/gpt3-model-sizes/
    "ada": 350_000_000,
    "babbage": 1_300_000_000,
    "curie": 6_700_000_000,
    "davinci": 175_000_000_000,
    # gpt neo sizes from their names
    "gpt-neo-125M": 125_000_000,
    "gpt-neo-1.3B": 1_300_000_000,
    "gpt-neo-2.7B": 2_700_000_000,
}

def main():
    args = parse_args(sys.argv[1:])
    project_dir = Path(__file__).resolve().parent.parent
    if args.colab:
        base_results_dir = Path("/content/drive/MyDrive/inverse_scaling_results")
    else:
        base_results_dir = Path(project_dir, "results")
    exp_dir = Path(base_results_dir, args.exp_dir)
    loss_csvs = [f for f in exp_dir.glob("*.csv") if f.name != "data.csv"]
    dfs = {csv_file.stem: pd.read_csv(csv_file, index_col=0) for csv_file in loss_csvs}

    averages = {model_name: np.mean(df["loss"]) for model_name, df in dfs.items()}
    standard_errors = {
        model_name: np.std(df["loss"]) / np.sqrt(len(df["loss"]))
        for model_name, df in dfs.items()
    }
    plot_loss(averages, standard_errors, exp_dir)


def plot_loss(loss_dict: dict[str, float], standard_errors: dict[str, float], exp_dir: Path) -> None:
    fig = plt.figure(figsize=(20, 10))
    errorbar_data = [
        (size_dict[size], loss, standard_errors[size])
        for size, loss in loss_dict.items()
    ]
    xs, ys, yerrs = zip(*sorted(errorbar_data, key=lambda pair: pair[0]))
    plt.errorbar(xs, ys, yerrs)

    labels, ticks = zip(
        *[
            (name, n_params)
            for name, n_params in size_dict.items()
            if name in loss_dict.keys()
        ]
    )
    plt.title("Log-log plot of classification loss vs model size")
    plt.xscale("log")
    plt.yscale("log")
    plt.xticks(ticks, labels, rotation=90)
    plt.legend()
    plt.savefig(Path(exp_dir, "loss_plot.png"))
    plt.show()


def parse_args(args) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot models in an experiment")
    parser.add_argument(
        "exp_dir",
        type=str,
        help="The name of the experiment to plot from",
    )
    parser.add_argument(
        "--colab",
        action="store_true",
        help="Whether to look for the exp dir in /content/drive or results",
    )
    args = parser.parse_args(args)
    return args


if __name__ == "__main__":
    main()
