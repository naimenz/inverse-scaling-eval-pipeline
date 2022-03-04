from __future__ import annotations
from abc import ABC, abstractmethod
import os
from typing import Union
from typing_extensions import Literal, get_args
from dotenv import load_dotenv
import requests
import torch
import torch.nn.functional as F
from datetime import timedelta
from ratelimit import limits, sleep_and_retry

from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
from eval_pipeline.dataset import Example

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


class HFModel(Model):
    def __init__(self, model_name: ValidHFModel, device: Device) -> None:
        self.device = device
        prefix = ""
        if model_name.startswith("gpt-neo") or model_name.startswith("gpt-j"):
            prefix = "EleutherAI/"
        self.model = AutoModelForCausalLM.from_pretrained(prefix + model_name).to(self.device)  # type: ignore
        self.tokenizer = AutoTokenizer.from_pretrained(prefix + model_name)

    def __call__(self, examples: list[Example]) -> list[float]:
        prompts = [example.prompt for example in examples]
        tokenized_inputs = self.tokenizer(prompts, return_tensors="pt").to(self.device)
        outputs = self.model(**tokenized_inputs)
        # we only need the logits for the final (new) token
        # NOTE: this may need to change if we use batch size > 1 with padding
        logits = outputs["logits"][:, -1]
        losses = self._losses_from_logits(examples, logits)
        return losses

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
    def __init__(self, model_name: ValidGPT3Model) -> None:
        self.model_name = model_name

    def __call__(self, examples: list[Example]) -> list[float]:
        losses = []
        for example in examples:
            choices = None
            # retries on failed calls, rate limiter should handle waiting
            while choices is None:
                response_json = self._call_api(example.prompt).json()
                choices = response_json.get("choices", None)
            logprobs = choices[0]["logprobs"]["top_logprobs"][0]
            relevant_logprobs = torch.Tensor([logprobs.get(c) for c in example.classes])
            loss = -F.log_softmax(relevant_logprobs, dim=-1)[example.answer_index]
            losses.append(loss.item())
        return losses

    @sleep_and_retry
    @limits(calls=60, period=timedelta(seconds=60).total_seconds())
    def _call_api(self, prompt: str) -> requests.Response:
        """This function makes the actual API call, and since we have a rate limit of 60 calls per minute,
        I will add rate limiting here (ideally we could increase the rate limit though)"""
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
        response = requests.post(url, json=data, headers=headers)
        return response
