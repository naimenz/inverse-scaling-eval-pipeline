from __future__ import annotations
import argparse
from datetime import datetime
import json
import sys
from typing import Union
import csv
import logging
import shutil
import pandas as pd
from pathlib import Path
import torch
from tqdm.autonotebook import tqdm

from eval_pipeline.dataset import Dataset
from eval_pipeline.models import Device, Model, ValidGPT3Model, ValidHFModel


def main():
    args = parse_args(sys.argv[1:])
    project_dir = Path(__file__).resolve().parent.parent
    base_data_dir = Path(project_dir, "data")
    # writing to google drive if we are in a colab notebook
    if args.colab:
        base_results_dir = Path("/content/drive/MyDrive/inverse_scaling_results/")
    else:
        base_results_dir = Path(project_dir, "results")

    if args.dataset_path is not None:
        data_path = args.dataset_path
    elif args.dataset is not None:
        data_path = Path(base_data_dir, args.dataset + ".csv")
    else:
        raise ValueError("One of --dataset or --dataset-path must be set")

    # if a results directory is supplied, use that as the experiment dir
    # otherwise, generate one from the dataset name and current time
    if args.exp_dir is not None:
        write_dir = Path(base_results_dir, args.exp_dir)
    else:
        current_time = datetime.now().replace(microsecond=0).isoformat()
        if args.dataset is not None:
            exp_dir = f"{current_time}_{args.dataset}"
        else:
            exp_dir = f"{current_time}_{Path(args.dataset_path).stem}"
        write_dir = Path(base_results_dir, exp_dir)
    write_dir.mkdir(parents=True, exist_ok=True)

    # we have to set up the logging AFTER deciding on a dir to write to
    log_path = Path(write_dir, "log.log")
    arg_log_path = Path(write_dir, "args.log")
    with arg_log_path.open("a") as f:
        json.dump(args.__dict__, f, indent=2)
    set_up_logging(log_path)

    logging.info(f"Logging set up with args\n{args}")
    logging.info(f"Saving to results to {write_dir}")

    # put a copy of the data in the experiment dir for reference
    shutil.copy(data_path, Path(write_dir, "data.csv"))
    logging.info("Copied data")
    data = load_data(data_path)

    device = "cuda:0" if args.use_gpu and torch.cuda.is_available() else "cpu"
    model_names = args.models
    for model_name in tqdm(model_names):
        run_model(model_name, data, write_dir, device)


def set_up_logging(log_path: Path):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )


def load_data(dataset_path: Path) -> Dataset:
    df = pd.read_csv(dataset_path, index_col=0)
    dataset = Dataset.from_df(df)
    return dataset


def run_model(
    model_name: Union[ValidHFModel, ValidGPT3Model],
    data: Dataset,
    write_dir: Path,
    device: Device,
):
    """This function needs to run the model on the data and
    write the results to write_path incrementally."""
    write_path = Path(write_dir, model_name + ".csv")
    field_names = ["index", "loss"]
    with write_path.open("w") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=field_names)
        writer.writeheader()
        model = Model.from_name(model_name, device)
        n_data = len(data)
        # TODO: Fix padding so I can use >1 batch size, and make it an input arg
        batch_size = 1
        for start_index in tqdm(range(0, n_data, batch_size)):
            examples = data.examples[start_index : start_index + batch_size]
            losses = model(examples)
            for offset, loss in enumerate(losses):
                writer.writerow({"index": start_index + offset, "loss": loss})


def parse_args(args):
    parser = argparse.ArgumentParser(
        description="Run model sizes and get losses for texts in a file"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="The name of the directory containing the data (must be a subdir of 'data'). Superseded by --dataset-path",
        required=False,
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        help="The path to the data file to use. Supersedes --dataset",
        required=False,
    )
    parser.add_argument(
        "--exp-dir",
        type=str,
        help="The name of the experiment to resume or write to",
        required=False,
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        help="The specific models to use",
        default=["gpt2", "gpt2-medium", "gpt2-large"],
        choices=[
            "gpt2",
            "gpt2-medium",
            "gpt2-large",
            "gpt2-xl",
            "gpt-neo-125M",
            "gpt-neo-1.3B",
            "gpt-neo-2.7B",
            "gpt-j-6B",
            "ada",
            "babbage",
            "curie",
            "davinci",
        ],
        required=True,
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Whether to use a GPU (if available)",
    )

    parser.add_argument(
        "--colab",
        action="store_true",
        help=(
            "Set if working in colab - will save results to gdrive mounted on /content/drive/MyDrive"
            " and use notebook versions of tqdm"
        ),
    )
    args = parser.parse_args(args)
    return args


if __name__ == "__main__":
    main()
