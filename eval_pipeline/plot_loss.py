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
    if args.task_type.startswith("classification") or args.task_type == "lambada":
        plot_classification_loss(
            exp_dir, args.dataset_sizes, args.task_type, args.invert
        )
    elif args.task_type == "numeric":
        plot_numeric_loss(exp_dir)
    else:
        raise ValueError(f"unknown task type {args.task_type}")


def plot_classification_loss(
    exp_dir: Path, dataset_sizes: list[int], task_type: TaskType, invert: bool
):
    loss_csvs = [f for f in exp_dir.glob("*.csv") if f.name != "data.csv"]
    data_csv = pd.read_csv(Path(exp_dir, "data.csv"), index_col=0).reset_index(
        drop=True
    )
    dfs = {csv_file.stem: pd.read_csv(csv_file, index_col=0) for csv_file in loss_csvs}

    if task_type == "classification_acc":
        # NOTE: assuming all examples have the same number of classes
        n_classes = len(literal_eval(data_csv["classes"][0]))  # type: ignore
        # the baseline puts equal probability on each class, so we are considering a uniform distribution
        baseline = 1 / n_classes
        output_name = "correct"
        if invert:
            for df in dfs.values():
                df.loc[:, output_name] = df[output_name].apply(
                    lambda correct: np.abs(correct - 1)
                )

    elif task_type == "classification_loss":
        # NOTE: assuming all examples have the same number of classes
        n_classes = len(literal_eval(data_csv["classes"][0]))  # type: ignore
        # the baseline puts equal probability on each class, so we are considering a uniform distribution
        baseline_prob = 1 / n_classes
        baseline = -np.log(baseline_prob)
        output_name = "loss"
        if invert:
            for df in dfs.values():
                df.loc[:, output_name] = df[output_name].apply(
                    lambda loss: -np.log(1 - np.exp(-loss))
                )

    else:
        baseline = None
        output_name = "loss"
        if invert:
            for df in dfs.values():
                df.loc[:, output_name] = df[output_name].apply(
                    lambda loss: -np.log(1 - np.exp(-loss))
                )

    if len(loss_csvs) == 0:
        raise ValueError(f"{exp_dir} does not exist or contains no output files")

    # one dict containing all different plots to be made, with their labels as keys
    separate_plot_dict = {}
    for size in dataset_sizes:
        size_dfs = {name: df[:size] for name, df in dfs.items()}
        averages = {
            model_name: np.mean(df[output_name]) for model_name, df in size_dfs.items()
        }
        # if invert:
        #     new_averages = dict()
        #     for name, loss in averages.items():
        #         new_loss =  -np.log(1 - np.exp(-loss))
        #         new_averages[name] = new_loss
        #     averages = new_averages
        standard_errors = {
            model_name: np.std(df[output_name]) / np.sqrt(len(df[output_name]))
            for model_name, df in size_dfs.items()
        }
        size_name = str(size) if size != -1 else len(list(dfs.values())[0])
        separate_plot_dict[size_name] = (averages, standard_errors)
    plot_loss(exp_dir, separate_plot_dict, baseline, task_type, invert)


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
    task_type: Optional[TaskType] = None,
    invert: Optional[bool] = None,
) -> None:
    plt.style.use("ggplot")

    fig = plt.figure(figsize=(6, 4))

    if baseline is not None:
        plt.axhline(
            baseline,
            linestyle="--",
            color="k",
            label="Baseline (equal probability)",
        )

    for label, (loss_dict, standard_errors) in separate_plots_dict.items():
        if standard_errors is not None:
            errorbar_data = [
                (size_dict[size], loss, standard_errors[size])
                for size, loss in loss_dict.items()
            ]
            xs, ys, yerrs = zip(*sorted(errorbar_data, key=lambda pair: pair[0]))
            plt.errorbar(
                xs, ys, yerrs, label=f"{label} examples (with Standard Error in Mean)"
            )
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

    if task_type == "classification_loss" or task_type == "lambada":
        plt.yscale("log")
        plt.ylabel("Loss")
        title = "Log-log plot of loss vs model size"
    elif task_type == "classification_acc":
        plt.ylabel("Accuracy")
        title = "Log plot of accuracy vs model size"
    else:
        raise ValueError
    if invert:
        title += " (inverted)"
    plt.title(title)
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
        default="classification_loss",
        choices=["classification_loss", "classification_acc", "numeric", "lambada"],
        help="The type of task to plot",
    )
    parser.add_argument(
        "--dataset-sizes",
        type=int,
        nargs="+",
        help="The numbers of examples to use (-1 means all)",
        default=[-1],
    )

    parser.add_argument(
        "--invert",
        action="store_true",
        help="Look at the loss on all OTHER class tokens (makes most sense for 2 classes)",
    )
    args = parser.parse_args(args)
    return args


if __name__ == "__main__":
    main()
