"""This function is supposed to take an experiment directory that ran on
anchoring and compute some loss for it.  Ideally I can abstract some of this
away and make a more general numeric evaluation script, but I'm not sure yet."""
from __future__ import annotations
import argparse
import json
import sys
from typing import cast
import numpy as np
import pandas as pd
from pathlib import Path


def main():
    args = parse_args(sys.argv[1:])
    project_dir = Path(__file__).resolve().parent.parent
    if args.colab:
        base_results_dir = Path("/content/drive/MyDrive/inverse_scaling_results")
    else:
        base_results_dir = Path(project_dir, "results")
    exp_dir = Path(base_results_dir, args.exp_dir)
    estimate_csvs = [f for f in exp_dir.glob("*.csv") if f.name != "data.csv"]
    if len(estimate_csvs) == 0:
        raise ValueError(f"{exp_dir} does not exist or contains no output files")
    input_df = pd.read_csv(Path(exp_dir, "data.csv"))
    output_dfs = {
        csv_file.stem: pd.read_csv(csv_file, index_col=0) for csv_file in estimate_csvs
    }

    df = input_df
    for name, output_df in output_dfs.items():
        output_df = output_df.rename(columns={"estimate": name})
        df = pd.merge(df, output_df, left_index=True, right_index=True)
    print(df.info())
    # for now, assuming that the pattern is 'control, small anchor, big anchor'
    # TODO: clean up how this works so it isn't based on assumptions
    losses = dict()
    for model_name, output_df in output_dfs.items():
        normalised_losses = []
        for i in range(0, len(df), 3):
            control = cast(float, output_df.iloc[i]["estimate"])
            low_anchor = cast(float, output_df.iloc[i + 1]["estimate"])
            high_anchor = cast(float, output_df.iloc[i + 2]["estimate"])
            # TODO: normalise based on the control or the true answer? doing control for now
            normed_control = control / control
            normed_low = low_anchor / control
            normed_high = high_anchor / control
            # TODO: decide on a much better loss calc
            loss = (normed_high - normed_control)**2 + (normed_low - normed_control)**2
            normalised_losses.append(loss)
        losses[model_name] = np.mean(normalised_losses)
    # writing as json for now
    with Path(exp_dir, "results.json").open("w") as f:
        json.dump(losses, f)


def parse_args(args) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute losses in an anchoring experiment"
    )
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
