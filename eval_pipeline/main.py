"""Take in a data file and a list of sizes and write out the losses for each text in the data file for each size"""
from __future__ import annotations
import argparse
import ast
import csv
from typing import Any, cast
import pandas as pd
from tqdm import tqdm

from eval_pipeline.hf_models import HFSize, HFWrapper
from eval_pipeline.gpt3 import GPT3Size, evaluate_gpt3_text


def main(args: argparse.Namespace):
    df = pd.read_csv(args.read_path, index_col=0)
    sizes = args.sizes
    hf_sizes = cast(
        "list[HFSize]", [size for size in sizes if size not in ("ada", "babbage", "curie", "davinci")]
    )
    gpt3_sizes = cast(
        "list[GPT3Size]", [size for size in sizes if size in ("ada", "babbage", "curie", "davinci")]
    )
    hf_models = {size: HFWrapper(size) for size in hf_sizes}

    with open(args.write_path, "w") as f:
        writer = csv.DictWriter(f, fieldnames=["text"] + hf_sizes + gpt3_sizes)
        writer.writeheader()
        for _, row in tqdm(df.iterrows(), total=df.shape[0]):
            row_dict = process_row(row, hf_models, gpt3_sizes)
            writer.writerow(row_dict)


def process_row(row, hf_models: dict[str, HFWrapper], gpt3_sizes: list[GPT3Size]):
    text = cast(str, (row["filled_template"]))
    answer_ix = cast(int, (row["answer_ix"]))
    # we need to convert the string back into a list by eval
    possible_answers = ast.literal_eval(row["possible_answers"])

    hf_loss_dict: dict[str, Any] = dict()
    for size, model in hf_models.items():
        loss = model.get_loss(text, answer_ix, possible_answers)
        hf_loss_dict[size] = loss

    gpt3_loss_dict: dict[str, Any] = evaluate_gpt3_text(
        text, gpt3_sizes, answer_ix, possible_answers
    )
    row_dict = {"text": text, **hf_loss_dict, **gpt3_loss_dict}
    return row_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run model sizes and get losses for texts in a file"
    )
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
    parser.add_argument(
        "--sizes",
        type=str,
        nargs="+",
        help="The specific model sizes to use",
        default=["gpt2", "ada", "babbage"],
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
    args = parser.parse_args()
    main(args)
