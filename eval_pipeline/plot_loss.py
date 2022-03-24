from __future__ import annotations
import argparse
from ast import literal_eval
import json
import sys

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from eval_pipeline.dataset import TaskType

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
    "gpt-j-6B": 6_000_000_000,
}


def main():
    args = parse_args(sys.argv[1:])
    project_dir = Path(__file__).resolve().parent.parent
    if args.colab:
        base_results_dir = Path("/content/drive/MyDrive/inverse_scaling_results")
    else:
        base_results_dir = Path(project_dir, "results")
    exp_dir = Path(base_results_dir, args.exp_dir)
    if args.task_type == "classification":
        plot_classification_loss(exp_dir, args.dataset_sizes)
    elif args.task_type == "numeric":
        plot_numeric_loss(exp_dir)
    else:
        raise ValueError(f"unknown task type {args.task_type}")


def plot_classification_loss(exp_dir: Path, dataset_sizes: list[int]):
    loss_csvs = [f for f in exp_dir.glob("*.csv") if f.name != "data.csv"]
    data_csv = pd.read_csv(Path(exp_dir, "data.csv"), index_col=0).reset_index(
        drop=True
    )
    # NOTE: assuming all examples have the same number of classes
    try:
        n_classes = len(literal_eval(data_csv["classes"][0]))  # type: ignore
        # the baseline puts equal probability on each class, so we are considering a uniform distribution
        baseline_prob = 1 / n_classes
        baseline_loss = -np.log(baseline_prob)
    except KeyError:
        print("No 'classes' so skipping baseline")
        baseline_loss = None
    if len(loss_csvs) == 0:
        raise ValueError(f"{exp_dir} does not exist or contains no output files")
    dfs = {csv_file.stem: pd.read_csv(csv_file, index_col=0) for csv_file in loss_csvs}
    # one dict containing all different plots to be made, with their labels as keys
    separate_plot_dict = {}
    for size in dataset_sizes:
        size_dfs = {name: df[:size] for name, df in dfs.items()}
        averages = {model_name: np.mean(df["loss"]) for model_name, df in size_dfs.items()}
        standard_errors = {
            model_name: np.std(df["loss"]) / np.sqrt(len(df["loss"]))
            for model_name, df in size_dfs.items()
        }
        size_name = str(size) if size != -1 else len(list(dfs.values())[0])
        separate_plot_dict[size_name] = (averages, standard_errors)
    plot_loss(exp_dir, separate_plot_dict, baseline=baseline_loss)


def plot_numeric_loss(exp_dir: Path):
    data_file = Path(exp_dir, "results.json")
    if not data_file.is_file():
        # TODO make error message more general
        raise ValueError(
            f"{data_file} does not exist: please run evaluate_anchoring first"
        )
    with data_file.open("r") as f:
        averages = json.load(f)
    plot_loss(exp_dir, {"numeric": (averages, None)})


def plot_loss(
    exp_dir: Path,
    separate_plots_dict: dict[str, tuple[dict, Optional[dict]]],
    baseline: Optional[float] = None,
) -> None:
    plt.style.use("ggplot")

    fig = plt.figure(figsize=(6, 4))

    if baseline is not None:
        plt.axhline(
            baseline,
            linestyle="--",
            color="k",
            label="Baseline loss (equal probability)",
        )

    for label, (loss_dict, standard_errors) in separate_plots_dict.items():
        if standard_errors is not None:
            errorbar_data = [
                (size_dict[size], loss, standard_errors[size])
                for size, loss in loss_dict.items()
            ]
            xs, ys, yerrs = zip(*sorted(errorbar_data, key=lambda pair: pair[0]))
            plt.errorbar(xs, ys, yerrs, label=f"{label} examples (with Standard Error in Mean)")
        else:
            xy_pairs = [(size_dict[size], loss) for size, loss in loss_dict.items()]
            xs, ys = zip(*sorted(xy_pairs, key=lambda pair: pair[0]))
            plt.plot(xs, ys, label="f{label} examples")

    labels, ticks = zip(
        *[
            (name, n_params)
            for name, n_params in size_dict.items()
            if name in loss_dict.keys()  # type: ignore (there has to be at least one loss dict)
        ]
    )

    plt.xscale("log")
    plt.xlabel("Model size")
    plt.xticks(ticks, labels, rotation=45)

    plt.yscale("log")
    plt.ylabel("Loss")

    plt.title("Log-log plot of loss vs model size")
    plt.legend()
    plt.tight_layout()
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
    parser.add_argument(
        "--task-type",
        type=str,
        default="classification",
        choices=["classification", "numeric", "lambada"],
        help="The type of task that was run in this experiment",
    )
    parser.add_argument(
        "--dataset-sizes",
        type=int,
        nargs="+",
        help="The numbers of examples to use (-1 means all)",
        default=[-1],
    )
    args = parser.parse_args(args)
    return args


if __name__ == "__main__":
    main()
