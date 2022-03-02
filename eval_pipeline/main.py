"""Take in a data file and a list of sizes and write out the losses for each text in the data file for each size"""
from __future__ import annotations
import argparse
import ast
import csv
import logging
from pprint import pprint
import time
from typing import Any, cast
import pandas as pd
import torch
from tqdm import tqdm

from eval_pipeline.hf_models import HFSize, HFWrapper, evaluate_hf_texts
from eval_pipeline.gpt3 import GPT3Size, evaluate_gpt3_text, evaluate_gpt3_texts


def main(args: argparse.Namespace):
    sizes = args.sizes
    hf_sizes = cast(
        "list[HFSize]",
        [size for size in sizes if size not in ("ada", "babbage", "curie", "davinci")],
    )
    gpt3_sizes = cast(
        "list[GPT3Size]",
        [size for size in sizes if size in ("ada", "babbage", "curie", "davinci")],
    )
    device = "cuda:0" if args.use_gpu and torch.cuda.is_available() else "cpu"

    df = pd.read_csv(args.read_path, index_col=0)
    # we need to convert the string of possible answers back into a list with eval
    df["possible_answers"] = df["possible_answers"].map(lambda x: ast.literal_eval(x))

    texts = cast("list[str]", list(df["filled_template"]))
    possible_answers = cast("list[tuple[str, ...]]", list(df["possible_answers"]))
    answer_indices = cast("list[int]", list(df["answer_ix"]))
    inputs = list(zip(texts, possible_answers, answer_indices))

    hf_losses = evaluate_hf_texts(inputs, sizes=hf_sizes, device=device)
    gpt3_losses = evaluate_gpt3_texts(inputs, sizes=gpt3_sizes)

    # combining the two sets of losses together into a single dict
    if len(hf_losses) > 0 and len(gpt3_losses) > 0:
        all_losses = {text: {**hf_losses[text], **gpt3_losses[text]} for text in texts}
    elif len(hf_losses) == 0:
        all_losses = gpt3_losses
    elif len(gpt3_losses) == 0:
        all_losses = hf_losses
    else:
        raise ValueError("must pass some sizes")

    rows = [{"text": text, **losses} for text, losses in all_losses.items()]
    df = pd.DataFrame.from_records(rows)
    df.to_csv(args.write_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Run model sizes and get losses for texts in a file"
    )
    parser.add_argument(
        "--read-path",
        type=str,
        help="The file path (relative or absolute) to the data file",
        required=True,
    )
    parser.add_argument(
        "--write-path",
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
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Whether to use a GPU (if available)",
    )
    args = parser.parse_args()
    main(args)
