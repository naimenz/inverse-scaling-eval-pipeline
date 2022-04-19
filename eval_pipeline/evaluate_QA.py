"""This function is supposed to take an experiment directory that ran on
QA and compute some loss for it. Ideally I can abstract some of this
away and make a more general evaluation script, but I'm not sure yet."""
from __future__ import annotations
import argparse
import json
import sys
from typing import cast
import numpy as np
import pandas as pd
from pathlib import Path


def main():
    # NOTE: 
    desired_type = None
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
    print(input_df.info())
    output_dfs = {
        csv_file.stem: pd.read_csv(csv_file, index_col=0) for csv_file in estimate_csvs
    }

    df = input_df
    for name, output_df in output_dfs.items():
        output_df = output_df.rename(columns={"logodds": name})
        output_df = output_df.drop(columns=["correct", "total_logprob"])
        df = pd.merge(df, output_df, left_index=True, right_index=True)
    print(df.info())
    # for now, assuming that the pattern is 'unbiased, biased'
    # TODO: clean up how this works so it isn't based on assumptions
    losses = dict()
    for model_name, output_df in output_dfs.items():
        logodds_losses = []
        for i in range(0, len(output_df), 2):
            unbiased_logodds = cast(float, output_df.iloc[i]["logodds"])
            biased_logodds = cast(float, output_df.iloc[i + 1]["logodds"])
            answer_index = input_df.iloc[i]["answer_index"]
            type = input_df.iloc[i]["type"]
            # DEBUG: looking at only one type
            if desired_type == None or type == desired_type:
                logodds_difference = unbiased_logodds - biased_logodds
                # flip the order (and hence the sign) if the answer is "no"
                if answer_index == 1:
                    logodds_difference *= -1
                
                logodds_losses.append(logodds_difference)
        losses[model_name] = np.mean(logodds_losses)
        print(len(logodds_losses))
    # writing as json for now
    with Path(exp_dir, "results.json").open("w") as f:
        json.dump(losses, f)


def parse_args(args) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute losses in a QA experiment"
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
