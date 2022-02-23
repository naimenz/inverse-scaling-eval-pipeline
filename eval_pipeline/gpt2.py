"""Code to handle the GPT-2 side of evaluation.
Uses the HuggingFace implementations of GPT-2.
Currently uses CPU because speed is not yet a concern.
"""
from __future__ import annotations
from typing_extensions import Literal
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
from pprint import pprint
from eval_pipeline.utils import wrap_question
import logging


GPT2Size = Literal["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]


class GPT2Wrapper:
    token_pairs = [("ĠYes", "ĠNo")]

    def __init__(self, size: GPT2Size) -> None:
        self.model = GPT2LMHeadModel.from_pretrained(size)
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.id2token = {i: t for t, i in self.tokenizer.vocab.items()}

    def get_positive_prob(self, text: str) -> float:
        """Return the probability that the answer to the question is in the affirmative.
        For now, we just check the tokens for " Yes" and " No", but more sophisticated schemes are
        possible with e.g. averaging over pairs of answers"""
        logits = self.get_logit_dict(text)
        # TODO: replace this with a loop over token pairs
        positive_token, negative_token = self.token_pairs[0]
        positive_logit = logits[positive_token]
        negative_logit = logits[negative_token]
        positive_prob, negative_prob = torch.nn.functional.softmax(
            torch.Tensor([positive_logit, negative_logit]), dim=-1
        )
        return positive_prob.item()

    def get_loss(self, text: str, answer: str) -> float:
        logits = self.get_logits(text)
        # get the token id of the answer token
        answer_id = self.tokenizer(answer)["input_ids"][0]
        logprobs = torch.nn.functional.log_softmax(logits)
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
    text_answer_pairs: list[tuple[str ,str]], sizes: list[GPT2Size]
) -> dict[str, dict[str, float]]:
    logging.info("CALLED GPT2")
    model_dict = {size: GPT2Wrapper(size) for size in sizes}
    all_prob_dicts = dict()

    for text, answer in text_answer_pairs:
        # for now, just using yes/no questions
        prepped_text = wrap_question(text)
        prob_dict = dict()
        for size, model in model_dict.items():
            # prob = model.get_positive_prob(prepped_text)
            prob = model.get_loss(prepped_text, answer)
            prob_dict[size] = prob
        all_prob_dicts[text] = prob_dict
    return all_prob_dicts


def main():
    # model = GPT2Wrapper('gpt2-medium')
    # text = "Answer the following question with Yes or No:\nIs the earth round?\nAnswer:"
    # text = "Is the earth round?\nAnswer:"
    # print("Round", model.get_positive_prob(text))
    # text = "Answer the following question with Yes or No:\nIs the earth flat?\nAnswer:"
    # text = "Is the earth flat?\nAnswer:"
    # print("Flat", model.get_positive_prob(text))
    sizes = ["gpt2", "gpt2-medium", "gpt2-large"]
    texts = [
        "Are bananas blue?",
        "Are bananas yellow?",
    ]
    # texts = [
    #     "You just won on the roulette wheel. Should you play again?",
    #     "You just lost on the roulette wheel. Should you play again?",
    # ]
    pprint(evaluate_gpt2_texts(texts, sizes))


if __name__ == "__main__":
    main()
