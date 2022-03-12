from __future__ import annotations
from abc import ABC, abstractmethod
import os
from typing import Union, cast
from typing_extensions import Literal, get_args
from dotenv import load_dotenv
import torch
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
from eval_pipeline.dataset import (
    ClassificationExample,
    Example,
    NumericExample,
    TaskType,
)
from eval_pipeline.numeric_parser import BasicParser
from eval_pipeline.openai_api import APIParameters, BaseGPT3Model, call_api

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

valid_gpt3_models: tuple[BaseGPT3Model, ...] = get_args(BaseGPT3Model)

Device = Literal["cuda:0", "cpu"]


api_parameters_map: dict[TaskType, APIParameters] = {
    "classification": APIParameters(
        temperature=0,
        n=1,
        max_tokens=1,
        logprobs=100,
    ),
    "numeric": APIParameters(
        temperature=0.2,
        n=5,
        max_tokens=10,
        logprobs=None,
        stop=["\n"],
    ),
}


class Model(ABC):
    @abstractmethod
    def __call__(self, examples: list[Example], task_type: TaskType) -> list[float]:
        raise NotImplementedError("Abstract method")

    @staticmethod
    def from_name(
        model_name: Union[ValidHFModel, BaseGPT3Model], device: Device
    ) -> Model:
        if model_name in valid_hf_models:
            model = HFModel(model_name, device)
        elif model_name in valid_gpt3_models:
            model = GPT3Model(model_name)
        else:
            raise ValueError(f"Unrecognised model '{model_name}'")
        return model


class HFModel(Model):
    def __init__(self, model_name: ValidHFModel, device: Device) -> None:
        self.device = device
        prefix = ""
        if model_name.startswith("gpt-neo") or model_name.startswith("gpt-j"):
            prefix = "EleutherAI/"
        self.model = AutoModelForCausalLM.from_pretrained(prefix + model_name).to(self.device)  # type: ignore
        # DEBUG: trying to suppress warning about sequence length
        self.model.max_length = 1024
        self.tokenizer = AutoTokenizer.from_pretrained(prefix + model_name)

    def __call__(self, examples: list[Example], task_type: TaskType) -> list[float]:
        # TODO: remove this restriction
        if len(examples) > 1:
            raise ValueError(
                f"Batch size of {len(examples)} not currently supported for HF models: please use 1"
            )
        prompts = [example.prompt for example in examples]
        tokenized_inputs = self.tokenizer(prompts, return_tensors="pt").to(self.device)
        if task_type == "classification":
            rv = self._evaluate_classification(examples, tokenized_inputs)
        elif task_type == "numeric":
            rv = self._evaluate_numeric(examples, tokenized_inputs)
        return rv

    def _evaluate_classification(
        self, examples: list[Example], tokenized_inputs: dict
    ) -> list[float]:
        outputs = self.model(**tokenized_inputs)
        # we only need the logits for the final (new) token
        # NOTE: this may need to change if we use batch size > 1 with padding
        logits = outputs["logits"][:, -1]
        losses = self._losses_from_logits(examples, logits)
        return losses

    def _evaluate_numeric(
        self, examples: list[Example], tokenized_inputs: dict
    ) -> list[float]:
        parser = BasicParser()
        # NOTE: this may need to change if we use batch size > 1 with padding
        outputs = self.model.generate(
            **tokenized_inputs,
            do_sample=True,
            num_return_sequences=5,
            max_new_tokens=7,
            temperature=0.2,
            pad_token_id=50526,
        )
        full_completions = self.tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )
        # strip out the prompt NOTE: again we're assuming the batch_size is 1
        untrimmed_completions = [
            fc[len(examples[0].prompt) :] for fc in full_completions
        ]
        # dropping anything after a new line
        print(f"untrimmed completions: {untrimmed_completions}")
        completions = [comp.split("\n")[0] for comp in untrimmed_completions]
        print(f"completions: {completions}")
        floats = parser(completions)
        print(f"floats: {floats}")
        # for now, we'll just take the mean of valid outputs as the estimate
        valid_floats = [f for f in floats if f is not None]
        print(f"Number of valid floats: {len(valid_floats)} of total: {len(floats)}")
        if len(valid_floats) > 0:
            estimate = sum(valid_floats) / len(valid_floats)
        else:
            raise ValueError("No valid numbers returned")
        return [estimate]

    def _losses_from_logits(self, examples, logits) -> list[float]:
        """Given examples and logits for those examples,
        compute the classification loss for each example"""
        losses = []
        for i, example in enumerate(examples):
            example_logits = logits[i]
            # have to flatten this list for some reason
            class_tokens = [
                token[0] for token in self.tokenizer(list(example.classes))["input_ids"]
            ]
            # log_softmax just subtracts a constant, so repeated applications change nothing
            # and there is no point in taking logprobs before focusing on the relevant indices
            relevant_logits = example_logits[class_tokens]
            loss = -F.log_softmax(relevant_logits, dim=-1)[example.answer_index]
            losses.append(loss.item())
        return losses


class GPT3Model(Model):
    def __init__(self, model_name: BaseGPT3Model) -> None:
        self.model_name: BaseGPT3Model = model_name

    def __call__(self, examples: list[Example], task_type: TaskType) -> list[float]:
        prompts = [example.prompt for example in examples]
        api_params = api_parameters_map[task_type]
        response_json = call_api(prompts, self.model_name, api_params).json()

        if task_type == "classification":
            classification_examples = cast("list[ClassificationExample]", examples)
            rv = self._evaluate_classification(classification_examples, response_json)
        elif task_type == "numeric":
            numeric_examples = cast("list[NumericExample]", examples)
            rv = self._evaluate_numeric(numeric_examples, response_json)
        return rv

    def _evaluate_classification(
        self, examples: list[ClassificationExample], response_json: dict
    ) -> list[float]:
        losses = []
        choices = response_json["choices"]
        for i, example in enumerate(examples):
            logprobs = choices[i]["logprobs"]["top_logprobs"][0]
            try:
                relevant_logprobs = torch.Tensor(
                    [logprobs.get(c) for c in example.classes]
                )
            except TypeError:
                raise ValueError(
                    f"Not all of {example.classes} were returned as logprobs by OpenAI"
                )

            loss = -F.log_softmax(relevant_logprobs, dim=-1)[example.answer_index]
            losses.append(loss.item())
        return losses

    def _evaluate_numeric(
        self, examples: list[NumericExample], response_json: dict
    ) -> list[float]:
        estimates = []
        choices = response_json["choices"]

        # working out which completions correspond to which input examples
        n_samples = len(choices) / len(examples)
        assert n_samples == int(n_samples)
        n_samples = int(n_samples)
        # parser = GPT3Parser("text-curie-001")
        parser = BasicParser()

        for i, example in enumerate(examples):
            start = i * n_samples
            completions = [
                choice["text"] for choice in choices[start : start + n_samples]
            ]
            print(f"completions: {completions}")
            floats = parser(completions)
            print(f"floats: {floats}")
            # for now, we'll just take the mean of valid outputs as the estimate
            valid_floats = [f for f in floats if f is not None]
            print(
                f"Number of valid floats: {len(valid_floats)} of total: {len(floats)}"
            )
            estimate = sum(valid_floats) / len(valid_floats)
            estimates.append(estimate)

        return estimates
