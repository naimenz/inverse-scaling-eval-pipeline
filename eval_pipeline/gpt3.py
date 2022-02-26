"""Code to handle the GPT-3 side of evaluation.
Uses requests to make http requests to the OpenAI API rather than their python package.
API key is kept in a .env file for privacy.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Optional
from typing_extensions import Literal
from eval_pipeline.utils import YAxis
import requests
import os
from dotenv import load_dotenv
import numpy as np
from pprint import pprint
import logging
import torch
import torch.nn.functional as F


GPT3Size = Literal["ada", "babbage", "curie", "davinci"]
OPENAI_API_BASE_URL = "https://api.openai.com/v1/engines"
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID")


def call_gpt3(text: str, size: GPT3Size) -> dict:
    # making it a variable for debugging
    max_logprobs = 100
    data = {
        "prompt": text,
        "temperature": 0,
        "n": 1,
        "max_tokens": 1,
        "logprobs": max_logprobs,
    }

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        # "OpenAI-Organization": OPENAI_ORG_ID,
        "Content-Type": "application/json",
    }

    url = os.path.join(OPENAI_API_BASE_URL, size, "completions")
    response_json = requests.post(url, json=data, headers=headers).json()
    try:
        logprobs = response_json["choices"][0]["logprobs"]["top_logprobs"][0]
    except Exception as e:
        pprint(f"FAILURE: {response_json}")
        raise e
    if len(logprobs) < max_logprobs:
        print(f"=== len(logprobs) = {len(logprobs)} ===")
        pprint(data)
        pprint(response_json)
        print(f"======")
    return response_json


def json_to_loss(
    json: dict,
    answer_ix: int,
    possible_answers: tuple[str, str],
) -> float:
    logprobs = json["choices"][0]["logprobs"]["top_logprobs"][0]
    print(len(logprobs))
    possible_logprobs = [logprobs.get(pa) for pa in possible_answers]
    if any(pl is None for pl in possible_logprobs):
        raise ValueError(
            f"logprobs {logprobs} doesn't contain all possible answers {possible_answers}"
        )
    normalised_logprobs = F.log_softmax(torch.Tensor(possible_logprobs), dim=-1)
    return -normalised_logprobs[answer_ix].item()


def evaluate_gpt3_text(
    text: str,
    sizes: list[GPT3Size],
    answer_ix: int,
    possible_answers: tuple[str, str],
) -> dict[str, float]:
    return_dict = dict()
    for size in sizes:
        logging.info(f"RUNNING {size}")
        json = call_gpt3(text, size)
        value = json_to_loss(json, answer_ix, possible_answers)
        return_dict[size] = value
    return return_dict
