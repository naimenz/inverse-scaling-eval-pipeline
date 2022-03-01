"""Take in a file full of losses, average them for each model size, and plot the results"""
from __future__ import annotations
import argparse
from matplotlib import pyplot as plt
import numpy as np
from eval_pipeline.utils import size_dict

import pandas as pd


def main(args: argparse.Namespace):
    df = pd.read_csv(args.read_path, index_col=0)
    loss_df = df.drop(columns=["text"])
    averages = {col: np.mean(loss_df[col]) for col in loss_df.columns}
    standard_errors = {col: np.std(loss_df[col]) / np.sqrt(len(df)) for col in loss_df.columns}
    print(standard_errors, len(df))
    plot_loss(averages, standard_errors)


def plot_loss(loss_dict: dict[str, float], standard_errors: dict[str, float]) -> None:
    fig = plt.figure(figsize=(20, 10))
    xy_pairs = [(size_dict[size], loss) for size, loss in loss_dict.items()]
    errorbar_data = [(size_dict[size], loss, standard_errors[size]) for size, loss in loss_dict.items()]
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
    plt.savefig(args.write_path)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Average and plot losses")
    parser.add_argument(
        "--read_path",
        type=str,
        help="The file path (relative or absolute) to the data file",
        required=True,
    )
    parser.add_argument(
        "--write_path",
        type=str,
        help="The file path (relative or absolute) to write results to",
        required=True,
    )
    args = parser.parse_args()
    main(args)
