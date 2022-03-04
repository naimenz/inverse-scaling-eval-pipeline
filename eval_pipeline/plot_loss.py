from __future__ import annotations
import argparse
import sys

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    args = parse_args(sys.argv[1:])
    exp_dir = Path(args.exp_dir)
    loss_csvs = [f for f in exp_dir.glob("*.csv") if f != "data.csv"]

    dfs = {csv_file.stem: pd.read_csv(csv_file, index_col=0) for csv_file in loss_csvs}
    df = pd.DataFrame()
        
    averages = {col: np.mean(loss_df[col]) for col in loss_df.columns}
    standard_errors = {col: np.std(loss_df[col]) / np.sqrt(len(model_df)) for col in loss_df.columns}
    plot_loss(averages, standard_errors)



def plot_loss(loss_dict: dict[str, float], standard_errors: dict[str, float]) -> None:
    fig = plt.figure(figsize=(20, 10))
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

def parse_args(args) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot models in an experiment"
    )
    parser.add_argument(
        "--exp-dir",
        type=str,
        help="The name of the experiment to plot from",
        required=True,
    )
    # args = parser.parse_args()
    args = parser.parse_args(args)
    return args

if __name__ == "__main__":
    main()