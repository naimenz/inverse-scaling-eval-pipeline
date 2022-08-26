from __future__ import annotations
from dataclasses import asdict, dataclass
import logging
import os
from typing_extensions import Literal
from dotenv import load_dotenv
import requests
from datetime import timedelta
from typing import Optional, Union
from ratelimit import sleep_and_retry, limits


BaseGPT3Model = Literal[
    "ada",
    "babbage",
    "curie",
    "davinci",
]
InstructGPT3Model = Literal[
    "text-ada-001",
    "text-babbage-001",
    "text-curie-001",
    "text-davinci-001",
]

OpenAIModel = Literal[BaseGPT3Model, InstructGPT3Model]


@dataclass
class APIParameters:
    temperature: float = 0.0
    n: int = 1
    max_tokens: int = 1
    top_p: float = 1.0
    logprobs: Optional[int] = 100
    stop: Optional[list[str]] = None
    echo: bool = False


OPENAI_API_BASE_URL = "https://api.openai.com/v1/engines"
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def call_api(
    prompt: Union[str, list[str]],
    model_name: OpenAIModel,
    api_params: Optional[APIParameters] = None,
) -> requests.Response:
    # dodgy error handling and retry code
    count = 0
    max_retries = 25
    while True:
        count += 1
        if count >= max_retries:
            raise ValueError(f"Retried too many times ({max_retries}), got error: {response_json['error']}")
        response = _call_api(prompt, model_name, api_params)
        response_json = response.json()
        if response.status_code != 200:
            logging.info(
                f"Retrying after error {response.status_code}: {response_json['error']}"
            )
        else:
            break
    return response


@sleep_and_retry
@limits(calls=50, period=timedelta(seconds=60).total_seconds())
def _call_api(
    prompt: Union[str, list[str]],
    model_name: OpenAIModel,
    api_params: Optional[APIParameters] = None,
) -> requests.Response:
    """This function makes the actual API call, and since we have a rate limit of 60 calls per minute,
    I will add rate limiting here (ideally we could increase the rate limit though)"""
    if api_params is None:
        api_params = APIParameters()
    # OpenAI gave my (Ian's) account the top 100 logprobs,
    # not just the top 5
    data = {
        "prompt": prompt,
        **asdict(api_params),
    }

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    url = os.path.join(OPENAI_API_BASE_URL, model_name, "completions")
    response = requests.post(url, json=data, headers=headers)
    return response
