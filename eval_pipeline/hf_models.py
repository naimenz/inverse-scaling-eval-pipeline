"""Code to handle the GPT-2 side of evaluation.
Uses the HuggingFace implementations of GPT-2.
Currently uses CPU because speed is not yet a concern.
"""
from __future__ import annotations
import json
import time
from typing import Iterable, Sequence
from typing_extensions import Literal
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
import torch
import torch.nn.functional as F
from pprint import pprint
import logging


HFSize = Literal[
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
    "gpt-neo-125M",
    "gpt-neo-1.3B",
    "gpt-neo-2.7B",
    "gpt-j-6B",
]


class HFWrapper:
    def __init__(self, size: HFSize, device: str = "cpu") -> None:
        # have to append the hoster if using Eleuther models
        self.device = device
        prefix = ""
        if size.startswith("gpt-neo") or size.startswith("gpt-j"):
            prefix = "EleutherAI/"
        self.model = AutoModelForCausalLM.from_pretrained(prefix + size).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(prefix + size)
        self.id2token = {i: t for t, i in self.tokenizer.vocab.items()}

    def get_loss(
        self, text: str, possible_answers: tuple[str, str], answer_ix: int
    ) -> float:
        """answer_ix gives the index of the intended answer from possible_answers"""
        logits = self.get_logits(text)
        # get the token id of the answer token
        positive_token_id, negative_token_id = self.tokenizer(possible_answers)[
            "input_ids"
        ]
        logprobs = F.log_softmax(logits, dim=-1)
        positive_logprob = logprobs[positive_token_id[0]]
        negative_logprob = logprobs[negative_token_id[0]]
        # DEBUG: checking alternative token choices
        other_pos, other_neg = self.tokenizer(
            [
                " 1",
                " 2",
            ]
        )["input_ids"]

        # For now I'm doing two log_softmaxes, which seems like it must be avoidable
        normalised_logprobs = F.log_softmax(
            torch.Tensor([positive_logprob, negative_logprob]), dim=-1
        )
        return -normalised_logprobs[answer_ix].item()

    def get_logits(self, text: str) -> torch.Tensor:
        encoded_input = self.tokenizer(text, return_tensors="pt").to(self.device)
        output = self.model(**encoded_input)
        raw_logits = output["logits"][0, -1]
        return raw_logits

    def get_logit_dict(self, text: str) -> dict[str, float]:
        logits = self.get_logits(text)
        logit_dict = {self.id2token[i]: logit for i, logit in enumerate(logits)}
        return logit_dict


def evaluate_hf_texts(
    text_possible_answers_ix_tuple: Iterable[tuple[str, tuple[str, ...], int]],
    sizes: Iterable[HFSize],
    device: str = "cpu",
) -> dict[str, dict[str, float]]:
    logging.info("CALLED HF")
    print(device)
    text_model_losses = {text: dict() for (text, _, _) in text_possible_answers_ix_tuple}
    for size in tqdm(sizes):
        tic = time.perf_counter()
        model = HFWrapper(size, device)
        toc = time.perf_counter()
        logging.info(f"Loaded {size} in {toc - tic:0.4f} seconds")
        for text, possible_answers, answer_ix in tqdm(text_possible_answers_ix_tuple, leave=False):
            value = model.get_loss(text, possible_answers, answer_ix)
            text_model_losses[text][size] = value
        # free the device memory before using the next size
        del model
        torch.cuda.empty_cache()
        # in case of crashes during evaluation, cache it for potential recovery
        with open('temp.cache', 'w') as f:
            json.dump(text_model_losses, f)
    return text_model_losses
