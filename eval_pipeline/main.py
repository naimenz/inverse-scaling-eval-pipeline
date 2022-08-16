from __future__ import annotations
import argparse
from datetime import datetime
import json
import sys
from typing import Union, cast
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
    base_results_dir = Path(project_dir, "results")

    if args.dataset_path is not None:
        data_path = Path(args.dataset_path)
    elif args.dataset is not None:
        data_path = Path(base_data_dir, args.dataset + ".csv")
    else:
        raise ValueError("One of --dataset or --dataset-path must be set")

    # if a results directory is supplied, use that as the experiment dir
    # otherwise, generate one from the dataset name and current time
    if args.exp_dir is not None:
        write_dir = Path(base_results_dir, args.exp_dir)
    else:
        write_dir = Path(".")
    write_dir.mkdir(parents=True, exist_ok=True)

    # we have to set up the logging AFTER deciding on a dir to write to
    log_path = Path(write_dir, "log.log")
    arg_log_path = Path(write_dir, "args.log")
    with arg_log_path.open("w") as f:
        json.dump(args.__dict__, f, indent=2)
    set_up_logging(log_path, args.logging_level)

    logging.info(f"Logging set up with args\n{args}")
    logging.info(f"Saving to results to {write_dir}")

    # put a copy of the data in the experiment dir for reference
    try:
        shutil.copy(data_path, Path(write_dir, f"data{data_path.suffix}"))
    except shutil.SameFileError:
        pass
    logging.info("Copied data")
    data = load_data(data_path, args.task_type)

    device = "cuda:0" if args.use_gpu and torch.cuda.is_available() else "cpu"
    model_names = args.models
    for model_name in tqdm(model_names):
        run_model(model_name, data, write_dir, device, args.batch_size, args.task_type)

    # final step to add all results to a jsonl
    labelled_df = load_df(data_path)
    for model_name in model_names:
        results_path = Path(write_dir, model_name + ".csv")
        prefix = f"{model_name}_"
        results = pd.read_csv(results_path, index_col=0)
        prefixed_results = results.add_prefix(prefix)
        labelled_df = labelled_df.merge(
            prefixed_results, left_index=True, right_index=True
        )
    labelled_path = Path(write_dir, "labelled_data.jsonl")
    labelled_df.to_json(labelled_path, orient="records", lines=True)


def set_up_logging(log_path: Path, logging_level: str):
    logging_levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warn": logging.WARN,
        "error": logging.ERROR,
    }
    
    logging.basicConfig(
        level=logging_levels[logging_level],
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    # suppress debug warnings from the Requests library
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def load_data(dataset_path: Path, task_type: TaskType) -> Dataset:
    df = load_df(dataset_path)
    if task_type in ["classification_loss", "classification_acc", "classification"]:
        dataset = Dataset.classification_from_df(df)
    elif task_type == "numeric":
        dataset = Dataset.numeric_from_df(df)
    elif task_type == "sequence_prob":
        dataset = Dataset.sequence_prob_from_df(df)
    elif task_type == "logodds" or task_type == "absolute_logodds":
        # we can just reuse the classification dataset type
        dataset = Dataset.logodds_from_df(df)
    else:
        raise ValueError(f"Unrecognised task type {task_type}")
    return dataset


def load_df(path: Path):
    if path.suffix == ".csv":
        return pd.read_csv(path)
    elif path.suffix == ".jsonl":
        return cast("pd.DataFrame", pd.read_json(path, lines=True))
    else:
        raise ValueError(f"Unknown file extension {path.suffix}")


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
    if task_type in ["classification_loss", "classification_acc", "classification"]:
        field_names = ["index", "loss", "correct", "predicted", "total_logprob"]
    elif task_type == "sequence_prob":
        field_names = ["index", "loss"]
    elif task_type == "numeric":
        field_names = ["index", "estimate"]
    elif task_type in ["logodds", "absolute_logodds"]:
        field_names = ["index", "logodds_difference", "correct", "total_logprob"]
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
                    rows[offset][output_name] = value
            for row in rows:
                writer.writerow(row)
        # DEBUG: trying to remove all references to the model so I can free GPU memory
        del model


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
            "text-ada-001",
            "text-babbage-001",
            "text-curie-001",
            "text-davinci-001",
            "opt-125m",
            "opt-350m",
            "opt-1.3b",
            "opt-2.7b",
            "opt-6.7b",
            "opt-13b",
        ],
        required=True,
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Whether to use a GPU (if available)",
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
        choices=[
            "classification",
            "classification_loss",
            "classification_acc",
            "sequence_prob",
            "logodds",
            "absolute_logodds",
        ],
    )
    parser.add_argument(
        "--logging-level",
        type=str,
        help="The level of logging to print",
        default="info",
        choices=[
            "debug",
            "info",
            "warn",
            "error",
        ],
    )
    args = parser.parse_args(args)
    return args


if __name__ == "__main__":
    main()
