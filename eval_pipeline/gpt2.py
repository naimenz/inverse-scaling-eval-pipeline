"""Code to handle the GPT-2 side of evaluation.
Uses the HuggingFace implementations of GPT-2.
Currently uses CPU because speed is not yet a concern.
"""
from __future__ import annotations
from typing_extensions import Literal
from transformers import GPT2LMHeadModel, GPT2TokenizerFast  # type: ignore
import torch
import torch.nn.functional as F
from pprint import pprint
from eval_pipeline.utils import YAxis, wrap_question
import logging


GPT2Size = Literal["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]


class GPT2Wrapper:
    def __init__(self, size: GPT2Size) -> None:
        self.model = GPT2LMHeadModel.from_pretrained(size)
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.id2token = {i: t for t, i in self.tokenizer.vocab.items()}

    def get_positive_prob(self, text: str, possible_answers: tuple[str, str]) -> float:
        """Return the probability that the answer to the question is in the affirmative.
        For now, we just check the tokens for " Yes" and " No", but more sophisticated schemes are
        possible with e.g. averaging over pairs of answers"""
        logits = self.get_logits(text)
        # TODO: replace this with a loop over token pairs
        positive_token_id, negative_token_id = self.tokenizer(list(possible_answers))[
            "input_ids"
        ]
        positive_logit = logits[positive_token_id[0]]
        negative_logit = logits[negative_token_id[0]]
        positive_prob, negative_prob = F.softmax(
            torch.Tensor([positive_logit, negative_logit]), dim=-1
        )
        return positive_prob.item()

    def get_loss(
        self, text: str, answer_ix: int, possible_answers: tuple[str, str]
    ) -> float:
        """answer_ix gives the index of the intended answer from possible_answers"""
        logits = self.get_logits(text)
        # get the token id of the answer token
        assert(possible_answers == [" Yes", " No"])
        positive_token_id, negative_token_id = self.tokenizer(possible_answers)[
            "input_ids"
        ]
        logprobs = F.log_softmax(logits, dim=-1)
        # TODO: replace this with a loop over token pairs
        positive_logprob = logprobs[positive_token_id[0]]
        negative_logprob = logprobs[negative_token_id[0]]

        # For now I'm doing two log_softmaxes, which seems like it must be avoidable
        normalised_logprobs = F.log_softmax(
            torch.Tensor([positive_logprob, negative_logprob]), dim=-1
        )
        return -normalised_logprobs[answer_ix].item()

    def get_logits(self, text: str) -> torch.Tensor:
        encoded_input = self.tokenizer(text, return_tensors="pt")
        output = self.model(**encoded_input)
        raw_logits = output["logits"][0, -1]
        return raw_logits

    def get_logit_dict(self, text: str) -> dict[str, float]:
        logits = self.get_logits(text)
        logit_dict = {self.id2token[i]: logit for i, logit in enumerate(logits)}
        return logit_dict


def evaluate_gpt2_texts(
    text_answer_ix_pairs: list[tuple[str, int]],
    sizes: tuple[GPT2Size, ...],
    y_axis: YAxis,
    possible_answers: tuple[str, str],
) -> dict[str, dict[str, float]]:
    logging.info("CALLED GPT2")
    model_dict = {size: GPT2Wrapper(size) for size in sizes}
    all_return_dicts = dict()

    for text, answer_ix in text_answer_ix_pairs:
        # for now, just using yes/no questions
        prepped_text = wrap_question(text)
        return_dict = dict()
        for size, model in model_dict.items():
            logging.info(f"RUNNING {size}")
            if y_axis == "positive_prob":
                value = model.get_positive_prob(prepped_text, possible_answers)
            elif y_axis == "loss":
                value = model.get_loss(prepped_text, answer_ix, possible_answers)
            return_dict[size] = value
        all_return_dicts[text] = return_dict
    return all_return_dicts
