from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator

import pandas as pd


@dataclass
class Example:
    prompt: str
    classes: tuple[str, ...]
    answer_index: int


class Dataset:
    """Class to store examples to be run by HF or GPT3 models"""

    def __init__(self, examples: list[Example]) -> None:
        self.examples = examples

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> Dataset:
        examples = []
        for _, (prompt, classes, answer_index) in df.iterrows():
            example = Example(prompt, classes, answer_index)
            examples.append(example)
        return Dataset(examples)

    def __iter__(self) -> Iterator[Example]:
        return iter(self.examples)

    def __len__(self) -> int:
        return len(self.examples)

