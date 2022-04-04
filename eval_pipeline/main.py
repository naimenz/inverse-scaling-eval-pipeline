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

from eval_pipeline.dataset import Dataset, TaskType
from eval_pipeline.models import Device, Model, BaseGPT3Model, ValidHFModel


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
    data = load_data(data_path, args.task_type)

    device = "cuda:0" if args.use_gpu and torch.cuda.is_available() else "cpu"
    model_names = args.models
    for model_name in tqdm(model_names):
        run_model(model_name, data, write_dir, device, args.batch_size, args.task_type)


def set_up_logging(log_path: Path):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )


def load_data(dataset_path: Path, task_type: TaskType) -> Dataset:
    df = pd.read_csv(dataset_path, index_col=0)
    if task_type == "classification":
        dataset = Dataset.classification_from_df(df)
    elif task_type == "numeric":
        dataset = Dataset.numeric_from_df(df)
    elif task_type == "lambada":
        dataset = Dataset.lambada_from_df(df)
    elif task_type == "QA":
        # we can just reuse the classification dataset
        dataset = Dataset.classification_from_df(df)
    return dataset  # type: ignore (classification is covered)


def run_model(
    model_name: Union[ValidHFModel, BaseGPT3Model],
    data: Dataset,
    write_dir: Path,
    device: Device,
    batch_size: int,
    task_type: TaskType,
):
    """This function needs to run the model on the data and
    write the results to write_path incrementally."""
    write_path = Path(write_dir, model_name + ".csv")
    # TODO: find a way to avoid having to specify field names ahead of time
    if task_type == "classification":
        field_names = ["index", "loss", "correct", "total_logprob"]
    elif task_type == "lambada":
        field_names = ["index", "loss"]
    elif task_type == "numeric":
        field_names = ["index", "estimate"]
    elif task_type == "QA":
        field_names = ["index", "logodds", "correct", "total_logprob"]
    else:
        raise ValueError(f"unknown task type {task_type}")

    with write_path.open("w") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=field_names)
        writer.writeheader()
        model = Model.from_name(model_name, device)
        n_data = len(data)
        # TODO: Fix padding so I can use >1 batch size for transformers models as well
        for start_index in tqdm(range(0, n_data, batch_size)):
            examples = data.examples[start_index : start_index + batch_size]
            outputs = model(examples, task_type)
            # we don't always have a full batch so just use the length of the actual output rather than the batch size
            n_outputs = len(list(outputs.values())[0])
            rows = [{"index": start_index + offset} for offset in range(n_outputs)]
            for output_name, values in outputs.items():
                for offset, value in enumerate(values):
                    try:
                        rows[offset][output_name] = value
                    except Exception as e:
                        print(f"len(rows) = {len(rows)}")
                        print(f"offset = {offset}")
                        print(f"len(values) = {len(values)}")
                        raise e
            for row in rows:
                writer.writerow(row)


def parse_args(args):
    parser = argparse.ArgumentParser(
        description="Run model sizes and get losses for texts in a file"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="The name of the file containing the dataset (must be in the directory 'data'). Superseded by --dataset-path",
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
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Only change the inference batch size if using exclusively GPT-3 models (will break HuggingFace models)",
        default=1,
    )
    parser.add_argument(
        "--task-type",
        type=str,
        help="The type of output expected for the dataset",
        default="classification",
        choices=["classification", "numeric", "lambada", "QA"],
    )
    args = parser.parse_args(args)
    return args


if __name__ == "__main__":
    main()
