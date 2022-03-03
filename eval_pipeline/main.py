from __future__ import annotations
from abc import ABC, abstractmethod
import argparse
from dataclasses import dataclass
from datetime import datetime
import os
import sys
from typing import Iterator, Union
import csv
from dotenv import load_dotenv
import requests

import pandas as pd
from pathlib import Path
import torch
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

from typing_extensions import Literal, get_args

OPENAI_API_BASE_URL = "https://api.openai.com/v1/engines"
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


ValidHFModel = Literal[
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
    "gpt-neo-125M",
    "gpt-neo-1.3B",
    "gpt-neo-2.7B",
    "gpt-j-6B",
]
valid_hf_models: tuple[ValidHFModel, ...] = get_args(ValidHFModel)

ValidGPT3Model = Literal[
    "ada",
    "babbage",
    "curie",
    "davinci",
]
valid_gpt3_models: tuple[ValidGPT3Model, ...] = get_args(ValidGPT3Model)

Device = Literal["cuda:0", "cpu"]


def main():
    args = parse_args(sys.argv[1:])
    # get the project root
    project_dir = Path(__file__).resolve().parent.parent
    base_data_dir = Path(project_dir, "data")
    base_results_dir = Path(project_dir, "results")
    # if a data directory is supplied, use that
    # otherwise, generate one from the dataset name and current time
    if args.exp_dir is not None:
        write_dir = Path(base_results_dir, "results", args.exp_dir)
    else:
        current_time = my_date = datetime.now().replace(microsecond=0).isoformat()
        exp_dir = f"{current_time}_{args.dataset}"
        write_dir = Path(base_results_dir, "results", exp_dir)
        write_dir.mkdir(parents=True, exist_ok=False)
    data = load_data(Path(base_data_dir, args.dataset + ".csv"))
    device = "cuda:0" if args.use_gpu and torch.cuda.is_available() else "cpu"
    model_names = args.models
    for model_name in tqdm(model_names):
        run_model(model_name, data, write_dir, device)


def load_data(dataset_path: Path) -> Dataset:
    df = pd.read_csv(dataset_path)
    dataset = Dataset.from_df(df)
    return dataset


@dataclass
class Example:
    prompt: str
    classes: tuple[str, ...]
    answer_index: int


class Dataset:
    """Class to store examples to be run by HF or GPT3 models"""

    def __init__(self, examples: list[Example]) -> None:
        self.examples = examples

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> Dataset:
        examples = [Example("I think therefore I", (" 1", " 2"), 0)] * len(df)
        return Dataset(examples)
        raise NotImplementedError()

    def __iter__(self) -> Iterator[Example]:
        return iter(self.examples)
    
    def __len__(self) -> int:
        return len(self.examples)


class Model(ABC):
    @abstractmethod
    def __call__(self, examples: list[Example]) -> list[float]:
        raise NotImplementedError("Abstract method")

    @staticmethod
    def from_name(
        model_name: Union[ValidHFModel, ValidGPT3Model], device: Device
    ) -> Model:
        if model_name in valid_hf_models:
            model = HFModel(model_name, device)
        elif model_name in valid_gpt3_models:
            model = GPT3Model(model_name)
        else:
            raise ValueError(f"Unrecognised model '{model_name}'")
        return model


class DummyModel(Model):
    def __call__(self, examples: list[Example]) -> list[float]:
        return [0.0] * len(examples)


class HFModel(Model):
    def __init__(self, model_name: ValidHFModel, device: Device) -> None:
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def __call__(self, examples: list[Example]) -> list[float]:
        prompts = [example.prompt for example in examples]
        tokenized_inputs = self.tokenizer(prompts, return_tensors="pt").to(self.device)
        outputs = self.model(**tokenized_inputs)
        logits = outputs["logits"][:, -1]
        return logits


class GPT3Model(Model):
    def __init__(self, model_name: ValidGPT3Model) -> None:
        self.model_name = model_name

    def __call__(self, examples: list[Example]) -> list[float]:
        raise NotImplementedError("Still working on it")
        for example in examples:
            response_json = self._call_api(example.prompt)

    def _call_api(self, prompt: str) -> dict:
        # OpenAI gave my (Ian's) account the top 100 logprobs,
        # not just the top 5
        max_logprobs = 100
        data = {
            "prompt": prompt,
            "temperature": 0,
            "n": 1,
            "max_tokens": 1,
            "logprobs": max_logprobs,
        }

        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }

        url = os.path.join(OPENAI_API_BASE_URL, self.model_name, "completions")
        response_json = requests.post(url, json=data, headers=headers).json()
        return response_json


def run_model(
    model_name: Union[ValidHFModel, ValidGPT3Model],
    data: Dataset,
    write_dir: Path,
    device: Device,
):
    """This function needs to run the model on the data and write the results to write_path incrementally."""
    write_path = Path(write_dir, model_name + ".csv")
    field_names = ["index", "loss"]
    with write_path.open("w") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=field_names)
        writer.writeheader()
        # DEBUG: dummy model for now
        # model = DummyModel()
        model = Model.from_name(model_name, device)
        # TODO: batch the inputs
        n_data = len(data)
        batch_size = 3
        for index in tqdm(range(0, n_data, batch_size)):
            examples = data.examples[index: index + batch_size]
            loss = model(examples)
            # TODO: fix this writing to work with batches
            writer.writerow({"index": index, "loss": loss})


def parse_args(args):
    parser = argparse.ArgumentParser(
        description="Run model sizes and get losses for texts in a file"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="The name of the directory containing the data (must be a subdir of 'data')",
        required=True,
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
    # args = parser.parse_args()
    args = parser.parse_args(args)
    return args


if __name__ == "__main__":
    main()
