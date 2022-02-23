"""Code to handle the GPT-3 side of evaluation.
Uses requests to make http requests to the OpenAI API rather than their python package.
API key is kept in a .env file for privacy.
"""
from __future__ import annotations
from collections import defaultdict
from typing_extensions import Literal
from eval_pipeline.utils import wrap_question
import requests
import os
from dotenv import load_dotenv
import numpy as np
from pprint import pprint
import logging


GPT3Size = Literal["ada", "babbage", "curie", "davinci"]
OPENAI_API_BASE_URL = "https://api.openai.com/v1/engines"
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID")


def call_gpt3(text: str, size: GPT3Size) -> dict:

    data = {
        "prompt": text,
        "temperature": 0,
        "n": 1,
        "max_tokens": 1,
        "logprobs": 5,
    }

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        # "OpenAI-Organization": OPENAI_ORG_ID,
        "Content-Type": "application/json",
    }

    url = os.path.join(OPENAI_API_BASE_URL, size, "completions")
    response_json = requests.post(url, json=data, headers=headers).json()
    return response_json


def json_to_positive_prob(
    json: dict, positive_token: str = " Yes", negative_token: str = " No"
) -> float:
    logprobs = json["choices"][0]["logprobs"]["top_logprobs"][0]
    positive_logprob = logprobs.get(positive_token)
    negative_logprob = logprobs.get(negative_token)
    if positive_logprob is None or negative_logprob is None:
        raise ValueError(
            f"logprobs {logprobs} doesn't contain positive token {positive_token} or negative token {negative_token}"
        )
    unnormed_positive_prob, unnormed_negative_prob = np.exp(positive_logprob), np.exp(
        negative_logprob
    )
    positive_prob = unnormed_positive_prob / (
        unnormed_positive_prob + unnormed_negative_prob
    )
    return positive_prob


def json_to_loss(
    json: dict,
    answer: str,
) -> float:
    logprobs = json["choices"][0]["logprobs"]["top_logprobs"][0]
    logprob = logprobs.get(answer)
    if logprob is None:
        raise ValueError(f"logprobs {logprobs} doesn't contain answer token {answer}")
    return -logprob


def evaluate_gpt3_text(
    text: str, answer: str, sizes: list[GPT3Size]
) -> dict[str, float]:
    prob_dict = dict()
    prepped_text = wrap_question(text)
    for size in sizes:
        json = call_gpt3(prepped_text, size)
        positive_prob = json_to_loss(json, answer)
        prob_dict[size] = positive_prob
    return prob_dict


def evaluate_gpt3_texts(
    text_answer_pairs: list[tuple[str, str]], sizes: list[GPT3Size]
) -> dict[str, dict[str, float]]:
    logging.info("CALLED GPT3")
    all_prob_dicts = dict()
    for text, answer in text_answer_pairs:
        all_prob_dicts[text] = evaluate_gpt3_text(text, answer, sizes)
    return all_prob_dicts


if __name__ == "__main__":
    positive_token = " Yes"
    negative_token = " No"
    texts = ["Are bananas blue?", "Are bananas yellow?"]
    pprint(evaluate_gpt3_texts(texts, ["ada", "babbage"]))
