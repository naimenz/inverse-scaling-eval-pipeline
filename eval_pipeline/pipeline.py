from __future__ import annotations
from typing import Union
from eval_pipeline.gpt2 import evaluate_gpt2_texts, GPT2Size
from eval_pipeline.gpt3 import evaluate_gpt3_texts, GPT3Size
from eval_pipeline.utils import size_dict
from concurrent.futures import ThreadPoolExecutor
import logging


def evaluate_texts(
    texts,
    gpt2_sizes: list[GPT2Size] = ["gpt2", "gpt2-medium", "gpt2-large"],
    gpt3_sizes: list[GPT3Size] = ["ada", "babbage", "curie", "davinci"],
) -> dict[str, dict[str, float]]: 
    with ThreadPoolExecutor(max_workers=2) as executor:
        gpt2_dict = executor.submit(evaluate_gpt2_texts, texts, gpt2_sizes)
        gpt3_dict = executor.submit(evaluate_gpt3_texts, texts, gpt3_sizes)
    combined_dict = {**gpt2_dict.result(), **gpt3_dict.result()}
    return combined_dict

# def plot_positive_probs(positive_prob_dict) -> None:


def main():
    texts = ["Is water a liquid?", "Is water a solid?"]
    positive_prob_dict = evaluate_texts(texts)


if __name__ == "__main__":
    format = "(%(threadName)s) %(asctime)s | %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")
    logging.info("logger")
    main()
