"""Code to handle the GPT-2 side of evaluation.
Uses the HuggingFace implementations of GPT-2.
Currently uses CPU because speed is not yet a concern.
"""
from __future__ import annotations
from typing import Optional
from typing_extensions import Literal
from transformers import GPT2LMHeadModel, GPT2TokenizerFast  # type: ignore
import torch
import torch.nn.functional as F
from pprint import pprint
from eval_pipeline.utils import YAxis, wrap_question
import logging


GPT2Size = Literal["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]


class GPT2Wrapper:
    token_pairs = [("ĠYes", "ĠNo")]

    def __init__(
        self, size: GPT2Size, possible_answers: Optional[tuple[str, str]]
    ) -> None:
        self.model = GPT2LMHeadModel.from_pretrained(size)
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.id2token = {i: t for t, i in self.tokenizer.vocab.items()}
        if possible_answers is not None:
            self.token_pairs = [possible_answers]

    def get_positive_prob(self, text: str) -> float:
        """Return the probability that the answer to the question is in the affirmative.
        For now, we just check the tokens for " Yes" and " No", but more sophisticated schemes are
        possible with e.g. averaging over pairs of answers"""
        logits = self.get_logit_dict(text)
        # TODO: replace this with a loop over token pairs
        positive_token, negative_token = self.token_pairs[0]
        positive_logit = logits[positive_token]
        negative_logit = logits[negative_token]
        positive_prob, negative_prob = F.softmax(
            torch.Tensor([positive_logit, negative_logit]), dim=-1
        )
        return positive_prob.item()

    def get_loss(self, text: str, answer: str) -> float:
        logits = self.get_logits(text)
        # get the token id of the answer token
        answer_id = self.tokenizer(answer)["input_ids"][0]
        logprobs = F.log_softmax(logits, dim=-1)
        logprob = logprobs[answer_id]
        return -logprob.item()

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
    text_answer_pairs: list[tuple[str, str]],
    sizes: tuple[GPT2Size, ...],
    y_axis: YAxis,
    possible_answers: Optional[tuple[str, str]] = None,
) -> dict[str, dict[str, float]]:
    logging.info("CALLED GPT2")
    model_dict = {size: GPT2Wrapper(size, possible_answers) for size in sizes}
    all_return_dicts = dict()

    for text, answer in text_answer_pairs:
        # for now, just using yes/no questions
        prepped_text = wrap_question(text)
        return_dict = dict()
        for size, model in model_dict.items():
            logging.info(f"RUNNING {size}")
            if y_axis == "positive_prob":
                value = model.get_positive_prob(prepped_text)
            elif y_axis == "loss":
                value = model.get_loss(prepped_text, answer)
            return_dict[size] = value
        all_return_dicts[text] = return_dict
    return all_return_dicts