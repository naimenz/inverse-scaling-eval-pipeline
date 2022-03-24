from __future__ import annotations
from abc import ABC, abstractmethod
import logging
import os
from typing import Union, cast
from typing_extensions import Literal, get_args
from dotenv import load_dotenv
import numpy as np
import torch
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
from eval_pipeline.dataset import (
    ClassificationExample,
    Example,
    LambadaExample,
    NumericExample,
    TaskType,
)
from eval_pipeline.numeric_parser import BasicParser
from eval_pipeline.openai_api import APIParameters, BaseGPT3Model, call_api

OPENAI_API_BASE_URL = "https://api.openai.com/v1/engines"
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# DEBUG: counting errors
error_count = 0
# for checking how long the input is
tokenizer = AutoTokenizer.from_pretrained("gpt2")

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
        if task_type.startswith("classification"):
            rv = self._evaluate_classification(examples, task_type)
        elif task_type == "numeric":
            rv = self._evaluate_numeric(examples)
        elif task_type == "lambada":
            rv = self._evaluate_lambada(examples)
        return rv  # type: ignore (we cover all cases, mypy can't do logic with startswith)

    def _evaluate_classification(
        self,
        examples: list[Example],
        task_type: TaskType,
    ) -> Union[list[float], list[int]]:
        prompts = [example.prompt for example in examples]
        tokenized_inputs = self.tokenizer(
            prompts, return_tensors="pt", truncation=True
        ).to(self.device)
        outputs = self.model(**tokenized_inputs)
        # we only need the logits for the final (new) token
        # NOTE: this may need to change if we use batch size > 1 with padding
        logits = outputs["logits"][:, -1]
        if task_type.endswith("loss"):
            losses = self._losses_from_logits(examples, logits)
            return losses
        elif task_type.endswith("acc"):
            accuracies = self._accuracies_from_logits(examples, logits)
            return accuracies
        else:
            raise ValueError(f"TaskType {task_type} not understood")

    def _evaluate_lambada(self, examples: list[Example]) -> list[float]:
        # finding the target
        prompts = [example.prompt for example in examples]
        tokenized_inputs = self.tokenizer(
            prompts, return_tensors="pt", truncation=True
        ).to(self.device)

        target_words = [" " + prompt.split(" ")[-1] for prompt in prompts]
        target_token_lengths = [
            len(self.tokenizer(word)["input_ids"]) for word in target_words
        ]

        outputs = self.model(**tokenized_inputs)
        logits = outputs["logits"]

        losses = []
        for i in range(len(examples)):
            # we only need the logits for the final word
            tokens = tokenized_inputs["input_ids"][i]
            # we have to go back by one because we don't care about the logits for the predicted token
            word_logits = logits[i, -target_token_lengths[i] - 1 : -1]
            word_tokens = tokens[-target_token_lengths[i] :]
            logprobs = -F.log_softmax(word_logits, dim=-1)
            loss = sum([logprobs[i, token] for i, token in enumerate(word_tokens)])
            losses.append(loss.item())  # type: ignore (the sum is never empty so never just 0, always a tensor)
        return losses

    def _evaluate_numeric(self, examples: list[Example]) -> list[float]:
        prompts = [example.prompt for example in examples]
        tokenized_inputs = self.tokenizer(
            prompts, return_tensors="pt", truncation=True
        ).to(self.device)
        parser = BasicParser()
        # NOTE: this may need to change if we use batch size > 1 with padding
        outputs = self.model.generate(
            **tokenized_inputs,
            do_sample=True,
            num_return_sequences=10,
            max_new_tokens=7,
            temperature=0.5,
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
        completions = [comp.split("\n")[0] for comp in untrimmed_completions]
        floats = parser(completions)
        # for now, we'll just take the mean of valid outputs as the estimate
        valid_floats = [f for f in floats if f is not None]
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

    def _accuracies_from_logits(self, examples, logits) -> list[int]:
        """Given examples and logits for those examples,
        compute whether the predicted label is correct for each example"""
        labels_correct = []
        for i, example in enumerate(examples):
            example_logits = logits[i]
            # have to flatten this list for some reason
            class_tokens = [
                token[0] for token in self.tokenizer(list(example.classes))["input_ids"]
            ]
            # log_softmax just subtracts a constant, so repeated applications change nothing
            # and there is no point in taking logprobs before focusing on the relevant indices
            relevant_logits = example_logits[class_tokens]
            label_correct = int(np.argmax(relevant_logits.cpu().detach().numpy()) == example.answer_index)
            labels_correct.append(label_correct)
        return labels_correct


class GPT3Model(Model):
    def __init__(self, model_name: BaseGPT3Model) -> None:
        self.model_name: BaseGPT3Model = model_name

    def __call__(self, examples: list[Example], task_type: TaskType) -> list[float]:

        if task_type == "classification":
            classification_examples = cast("list[ClassificationExample]", examples)
            rv = self._evaluate_classification(classification_examples)
        elif task_type == "numeric":
            numeric_examples = cast("list[NumericExample]", examples)
            rv = self._evaluate_numeric(numeric_examples)
        elif task_type == "lambada":
            lambada_examples = cast("list[LambadaExample]", examples)
            rv = self._evaluate_lambada(lambada_examples)
        return rv

    def _evaluate_classification(
        self,
        examples: list[ClassificationExample],
    ) -> list[float]:
        prompts = [example.prompt for example in examples]
        api_params = APIParameters(
            temperature=0,
            n=1,
            max_tokens=1,
            logprobs=100,
        )
        response_json = call_api(prompts, self.model_name, api_params).json()
        losses = []
        choices = response_json["choices"]
        for i, example in enumerate(examples):
            logprobs = choices[i]["logprobs"]["top_logprobs"][0]
            try:
                relevant_logprobs = torch.Tensor(
                    [logprobs.get(c) for c in example.classes]
                )
            except TypeError:
                global error_count
                logging.info(f"error_count = {error_count}")
                logging.info(example)
                logging.info(logprobs)
                # DEBUG: not raising an error, just moving on to the next example
                error_count += 1
                continue
                # raise ValueError(
                #     f"Not all of {example.classes} were returned as logprobs by OpenAI"
                # )

            loss = -F.log_softmax(relevant_logprobs, dim=-1)[example.answer_index]
            losses.append(loss.item())
        return losses

    def _evaluate_lambada(self, examples: list[LambadaExample]) -> list[float]:
        prompts = [example.prompt for example in examples]
        api_params = APIParameters(
            temperature=0.0,
            n=1,
            max_tokens=0,
            logprobs=1,
            stop=["\n"],
            echo=True,
        )
        response_json = call_api(prompts, self.model_name, api_params).json()
        prompts = [example.prompt for example in examples]

        losses = []
        for i in range(len(examples)):
            text_index = len(prompts[i]) - 1 - prompts[i][::-1].index(" ")
            logprobs_dict = response_json["choices"][i]["logprobs"]
            text_offset = logprobs_dict["text_offset"]
            actual_logprobs = logprobs_dict["token_logprobs"]
            token_index = text_offset.index(text_index)

            loss = 0
            for logprob in actual_logprobs[token_index:]:
                loss -= logprob
            losses.append(loss)

        return losses

    def _evaluate_numeric(self, examples: list[NumericExample]) -> list[float]:
        prompts = [example.prompt for example in examples]
        api_params = APIParameters(
            temperature=0.5,
            n=10,
            max_tokens=10,
            logprobs=None,
            stop=["\n"],
        )
        response_json = call_api(prompts, self.model_name, api_params).json()
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
            floats = parser(completions)
            # for now, we'll just take the mean of valid outputs as the estimate
            valid_floats = [f for f in floats if f is not None]
            estimate = sum(valid_floats) / len(valid_floats)
            estimates.append(estimate)

        return estimates
