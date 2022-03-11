from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional

from eval_pipeline.openai_api import InstructGPT3Model, call_api, APIParameters

template = """
Q: Convert "2 million" to a number.
A: 2000000

Q: Convert "a whole lot" to a number.
A: N/A

Q: Convert "{number_string}" to a number.
A:
""".strip()

parser_api_params = APIParameters(max_tokens=10, stop=["\n"])


class NumericParser(ABC):
    @abstractmethod
    def __call__(self, number_strings: list[str]) -> list[Optional[float]]:
        pass


class BasicParser(NumericParser):
    def __call__(self, number_strings: list[str]) -> list[Optional[float]]:
        prepped_strings = [prep_string(s) for s in number_strings]
        floats_from_parsed_strings: list[Optional[float]] = []
        for s in prepped_strings:
            try:
                parsed_s = float(s)
                floats_from_parsed_strings.append(parsed_s)
            except ValueError:
                floats_from_parsed_strings.append(None)

        return floats_from_parsed_strings


def prep_string(s: str) -> str:
    # clearing out leading/trailing spaces, commas, parentheses, and trailing full stops
    return s.strip().replace(",", "").replace(")", "").replace("(", "").rstrip(".")


class GPT3Parser(NumericParser):
    def __init__(self, model_name: InstructGPT3Model) -> None:
        self.model_name: InstructGPT3Model = model_name

    def __call__(self, number_strings: list[str]) -> list[Optional[float]]:
        filled_templates = [
            template.format(number_string=ns.strip()) for ns in number_strings
        ]
        response = call_api(filled_templates, self.model_name, parser_api_params)
        texts = [
            choice["text"].strip().replace(",", "")
            for choice in response.json()["choices"]
        ]
        print(f"strings returned from parser: {texts}")
        # check the response is non-empty and the model reports it found a valid number
        floats = [float(text) if text and text != "N/A" else None for text in texts]
        return floats
